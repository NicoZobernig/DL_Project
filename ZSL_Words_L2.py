import torch
import torch.cuda
import numpy as np
import argparse

from torch.utils.data import DataLoader
from zsldataset import ZSLDataset
from models import ContinuousMap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--leonhard', type=bool, default=False)
    parser.add_argument('--use_irevnet', type=bool, default=True)  # use iRevNet features or ResNet101 features batch_size

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)

    parser.add_argument('--optimizer', type=str, default='sgd')

    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--alphas', nargs='+', type=float, default=[20, 1e-3, 1e-3])

    parser.add_argument('--margin', type=int, default=3)

    parser.add_argument('--gamma', type=float, default=0.3)

    parser.add_argument('--momentum', type=float, default=0.5)

    return parser.parse_args()


def dist_matrix(batch1, batch2):
    delta = batch1.unsqueeze(1) - batch2.unsqueeze(0)
    d_matrix = (delta * delta).mean(dim=-1)

    return d_matrix


def mag(u):
    return torch.dot(u, u)


def dist(u, v):
    return torch.dot(u - v, u - v)


def main():
    options = parse_args()

    # Load Data
    if options.leonhard:
        train_path = 'ZSL_Data/AwA2_train'
        test_path = 'ZSL_Data/AwA2_test'
    else:
        train_path = 'Data/AwA2/train_set'
        test_path = 'Data/AwA2/test_set'

    trainset = ZSLDataset(train_path, use_predicates=False, use_irevnet=options.use_irevnet)
    testset = ZSLDataset(test_path, use_predicates=False, use_irevnet=options.use_irevnet)

    num_classes = trainset.classes.shape[0]

    dim_semantic = trainset[0]['class_embedding'].shape[0]
    dim_visual = trainset[0]['image_embedding'].shape[0]

    all_class_embeddings = torch.tensor(np.array(trainset.class_embeddings)).cuda().float()
    classes_enum = torch.tensor(np.array(range(num_classes), dtype=np.int64)).cuda()

    query_ids = set([testset[i]['class_id'] for i in range(len(testset))])
    ids = list(i - 1 for i in query_ids)
    query_mask = np.zeros(num_classes)
    query_mask[ids] = 1
    query_mask = torch.tensor(query_mask, dtype=torch.int64).cuda()

    v_to_s = ContinuousMap(dim_source=dim_visual, dim_dest=dim_semantic, width=512).cuda()
    s_to_v = ContinuousMap(dim_source=dim_semantic, dim_dest=dim_visual, width=512).cuda()

    if options.optimizer == 'adam':
        optimizer = torch.optim.Adam(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                     lr=options.learning_rate,
                                     betas=(0.9, 0.999),
                                     weight_decay=3e-3)
    else:
        optimizer = torch.optim.SGD(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                    lr=options.learning_rate,
                                    momentum=options.momentum,
                                    weight_decay=5e-3,
                                    nesterov=True)

    trainloader = DataLoader(trainset,
                             batch_size=options.batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)

    testloader = DataLoader(testset,
                            batch_size=options.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=True)

    alpha1 = options.alphas[0]  # l2
    alpha2 = options.alphas[1]  # surjection
    alpha3 = options.alphas[2]  # l2 regularization

    # Main Loop
    for e in range(options.n_epochs):
        v_to_s = v_to_s.train()
        s_to_v = s_to_v.train()

        running_loss = 0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            batch_visual = sample['image_embedding'].cuda().float()
            batch_classes = sample['class_id'].cuda() - 1

            e_hat = v_to_s(s_to_v(all_class_embeddings))
            delta = (e_hat - all_class_embeddings)
            surjection_loss = (delta * delta).sum(dim=-1).mean()

            s_out = v_to_s(batch_visual)

            same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
            same_class = same_class.detach()

            l2_dist_loss = (dist_matrix(s_out, all_class_embeddings) * same_class.float()).mean()
            l2_loss = (s_out * s_out).sum(dim=-1).mean()
            loss = alpha1 * l2_dist_loss + alpha2 * surjection_loss + alpha3 * l2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Training Loss epoch {0}: {1}'.format(e, running_loss/len(trainloader)))

        if (e + 1) % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7

        if (e + 1) % 5 == 0:
            print('\n\n- Evaluation on epoch {}'.format(e+1))

            avg_accuracy = 0.
            avg_loss = 0.
            n = 0

            v_to_s = v_to_s.eval()
            s_to_v = s_to_v.eval()

            with torch.no_grad():
                for i, sample in enumerate(testloader):
                    n += 1

                    batch_visual = sample['image_embedding'].cuda().float()
                    batch_classes = sample['class_id'].cuda() - 1

                    s_out = v_to_s(batch_visual)

                    same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                    same_class = same_class.detach()

                    d_matrix = dist_matrix(s_out, all_class_embeddings)

                    c_hat = (d_matrix + (1 - query_mask).float() * 1e6).argmin(dim=-1)

                    l2_dist_loss = (dist_matrix(s_out, all_class_embeddings) * same_class.float()).mean()

                    loss = alpha1 * l2_dist_loss

                    avg_loss += loss.item()
                    avg_accuracy += (c_hat == batch_classes).float().mean().item()

            avg_accuracy /= n
            avg_loss /= n

            print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))


if __name__ == '__main__':
    main()
