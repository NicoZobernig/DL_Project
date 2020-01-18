import torch
import torch.cuda
import numpy as np
import argparse
import sys

from torch.utils.data import DataLoader
from zsldataset import ZSLDataset
from models import LinearDecoderAttributes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-only_words', action='store_const', const=True, default=False,
                        help='Only use class word embeddings')
    parser.add_argument('-only_attributes', action='store_const', const=True, default=False,
                        help='Only use class attributes')

    parser.add_argument('-use_resnet', action='store_const', const=True, default=False,
                        help='use ResNet101 image embeddings')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='sgd', help='\'sgd\'(default) or \'adam\'')

    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--alphas', nargs='+', type=float, default=[40, 1e-3, 1e-3],
                        help='weight parameters of loss metric - surjectivity - l2 regularization')
    parser.add_argument('--margin', type=int, default=3, help='margin of triplet loss')
    parser.add_argument('--gamma', type=float, default=0.3, help='mixture parameter of class embedding and attributes')
    parser.add_argument('--momentum', type=float, default=0.55)
    parser.add_argument('--weight_decay', type=float, default=3e-3)

    return parser.parse_known_args()


def dist_matrix(batch1, batch2):
    delta = batch1.unsqueeze(1) - batch2.unsqueeze(0)
    d_matrix = (delta * delta).mean(dim=-1)

    return d_matrix


def main():
    torch.manual_seed(1)
    data_path = sys.argv[1]
    options, _ = parse_args()

    train_path = data_path + 'train_set'
    test_path = data_path + 'test_set'

    trainset = ZSLDataset(train_path, use_irevnet=not options.use_resnet)
    testset = ZSLDataset(test_path, use_irevnet=not options.use_resnet)

    num_classes = trainset.classes.shape[0]
    classes_enum = torch.tensor(np.array(range(num_classes), dtype=np.int64)).cuda()

    dim_visual = trainset[0]['image_embedding'].shape[0]
    dim_semantic = trainset[0]['class_embedding'].shape[0]
    dim_attributes = trainset[0]['class_predicates'].shape[0]

    all_class_embeddings = torch.tensor(np.array(trainset.class_embeddings)).float().cuda()
    all_class_predicates = torch.tensor(np.array(trainset.class_predicates)).float().cuda()

    query_ids = set([testset[i]['class_id'] for i in range(len(testset))])
    ids = list(i - 1 for i in query_ids)
    query_mask = np.zeros(num_classes)
    query_mask[ids] = 1
    query_mask = torch.tensor(query_mask, dtype=torch.int64).cuda()

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

    v_to_s = LinearDecoderAttributes(dim_source=dim_visual,
                                     dim_target1=dim_attributes,
                                     dim_target2=dim_semantic,
                                     width=512).cuda()

    if options.optimizer == 'adam':
        optimizer = torch.optim.Adam(list(v_to_s.parameters()),
                                     lr=options.learning_rate,
                                     betas=(0.9, 0.999),
                                     weight_decay=3e-3)
    else:
        optimizer = torch.optim.SGD(list(v_to_s.parameters()),
                                    lr=options.learning_rate,
                                    momentum=options.momentum,
                                    weight_decay=options.weight_decay,
                                    nesterov=True)

    if options.only_words:
        gamma = 0
    elif options.only_attributes:
        gamma = 1
    else:
        gamma = options.gamma

    alpha1 = options.alphas[0]  # l2
    # alpha2 = options.alphas[1]
    alpha3 = options.alphas[2]  # l2 regularization

    for e in range(options.n_epochs):
        v_to_s = v_to_s.train()

        running_loss = 0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            batch_visual = sample['image_embedding'].cuda().float()

            batch_classes = sample['class_id'].cuda() - 1

            s_out = v_to_s(batch_visual)
            s_attr, s_word = s_out

            same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
            same_class = same_class.detach()

            d_matrix = (1 - gamma) * dist_matrix(s_word, all_class_embeddings) + \
                       gamma * dist_matrix(s_attr, all_class_predicates)

            l2_dist_loss = (d_matrix * same_class.float()).mean()
            l2_loss = (1 - gamma) * (s_word * s_word).sum(dim=-1).mean() + gamma * (s_attr * s_attr).sum(dim=-1).mean()
            loss = alpha1 * l2_dist_loss + alpha3 * l2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Training Loss epoch {0}: {1}'.format(e, running_loss / len(trainloader)))

        if (e + 1) % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7

    print('-----------------------------')
    print('\nEvaluation on test data: \n')

    avg_accuracy = 0.
    n = 0

    v_to_s = v_to_s.eval()

    with torch.no_grad():
        for i, sample in enumerate(testloader):
            n += 1

            batch_visual = sample['image_embedding'].float().cuda()
            batch_classes = sample['class_id'].cuda() - 1

            s_out = v_to_s(batch_visual)
            s_attr, s_word = s_out

            d_matrix = (1 - gamma) * dist_matrix(s_word, all_class_embeddings) + \
                       gamma * dist_matrix(s_attr, all_class_predicates)

            c_hat = (d_matrix + (1 - query_mask).float() * 1e9).argmin(dim=-1)

            avg_accuracy += (c_hat == batch_classes).float().mean().item()

    avg_accuracy /= n
    print('Accuracy: {0}'.format(avg_accuracy))
    print('-----------------------------')


if __name__ == '__main__':
    main()
