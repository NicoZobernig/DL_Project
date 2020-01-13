import torch
import torch.cuda
import numpy as np

from torch.utils.data import DataLoader
from zsldataset import ZSLDataset
from models import EncoderAttributes, DecoderAttributes

from ZSL_Words_L2 import parse_args, dist_matrix


def main():
    options = parse_args()

    # Load Data
    if options.leonhard:
        train_path = 'ZSL_Data/AwA2_train'
        test_path = 'ZSL_Data/AwA2_test'
    else:
        train_path = 'Data/AwA2/train_set'
        test_path = 'Data/AwA2/test_set'

    trainset = ZSLDataset(train_path, use_predicates=True, use_irevnet=options.use_irevnet)
    testset = ZSLDataset(test_path, use_predicates=True, use_irevnet=options.use_irevnet)

    num_classes = trainset.classes.shape[0]

    dim_semantic = trainset[0]['class_embedding'].shape[0]
    dim_visual = trainset[0]['image_embedding'].shape[0]
    dim_attributes = trainset[0]['class_predicates'].shape[0]

    all_class_embeddings = torch.tensor(np.array(trainset.class_embeddings)).float().cuda()
    all_class_predicates = torch.tensor(np.array(trainset.class_predicates)).float().cuda()
    classes_enum = torch.tensor(np.array(range(num_classes), dtype=np.int64)).cuda()

    query_ids = set([testset[i]['class_id'] for i in range(len(testset))])
    ids = list(i - 1 for i in query_ids)
    query_mask = np.zeros(num_classes)
    query_mask[ids] = 1
    query_mask = torch.tensor(query_mask, dtype=torch.int64).cuda()

    v_to_s = DecoderAttributes(dim_source=dim_visual,
                               dim_target1=dim_attributes,
                               dim_target2=dim_semantic,
                               width=512).cuda()

    s_to_v = EncoderAttributes(dim_source1=dim_semantic,
                               dim_source2=dim_attributes,
                               dim_target=dim_visual,
                               width=512).cuda()

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

    positive_part = torch.nn.ReLU().cuda()

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

    gamma = options.gamma

    alpha1 = options.alphas[0]  # triplet
    alpha2 = options.alphas[1]  # surjection
    alpha3 = options.alphas[2]  # l2 regularization

    margin = options.margin

    validation_accuracy = []
    for e in range(options.n_epochs):
        v_to_s = v_to_s.train()
        s_to_v = s_to_v.train()

        running_loss = 0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            batch_visual = sample['image_embedding'].float().cuda()

            batch_classes = sample['class_id'].cuda() - 1

            e_hat = v_to_s(s_to_v(all_class_embeddings, all_class_predicates))
            delta = (e_hat[1] - all_class_embeddings)
            surjection_loss = (delta * delta).sum(dim=-1).mean()
            delta = (e_hat[0] - all_class_predicates)
            surjection_loss = (1 - gamma) * surjection_loss + gamma * (delta * delta).sum(dim=-1).mean()

            s_out = v_to_s(batch_visual)
            s_attr, s_word = s_out

            same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
            same_class = same_class.detach()

            d_matrix = (1 - gamma) * dist_matrix(s_word, all_class_embeddings) + gamma * dist_matrix(s_attr, all_class_predicates)

            closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
            furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

            l2_loss = (1 - gamma) * (s_word * s_word).sum(dim=-1).mean() + gamma * (s_attr * s_attr).sum(dim=-1).mean()
            loss = positive_part(furthest_positive - closest_negative + margin)
            loss = alpha1 * loss.mean() + alpha2 * surjection_loss + alpha3 * l2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Training Loss epoch {0}: {1}'.format(e+1, running_loss/len(trainloader)))

        if (e + 1) % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

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

                    batch_visual = sample['image_embedding'].float().cuda()
                    batch_classes = sample['class_id'].cuda() - 1

                    s_out = v_to_s(batch_visual)
                    s_attr, s_word = s_out

                    same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                    same_class = same_class.detach()

                    d_matrix = (1 - gamma) * dist_matrix(s_word, all_class_embeddings) + gamma * dist_matrix(s_attr, all_class_predicates)

                    c_hat = (d_matrix + (1 - query_mask).float() * 1e9).argmin(dim=-1)

                    closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                    furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                    loss = alpha1 * furthest_positive.mean()

                    avg_loss += loss.item()
                    avg_accuracy += (c_hat == batch_classes).float().mean().item()

            avg_accuracy /= n
            avg_loss /= n

            if e > 50:
                validation_accuracy.append(avg_accuracy)

            print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))

    print('Mean Accuracy: {0}'.format(np.mean(validation_accuracy)))


if __name__ == '__main__':
    main()
