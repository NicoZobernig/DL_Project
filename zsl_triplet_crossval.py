import torch
import torch.cuda
import numpy as np
import argparse
import sys

from torch.utils.data import DataLoader
from zsldataset import ZSLDataset
from models import ContinuousMap, EncoderAttributes, DecoderAttributes


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
    validation_path = data_path + 'val_set'

    train_set_1 = ZSLDataset(train_path+'_1', use_irevnet=not options.use_resnet)
    train_set_2 = ZSLDataset(train_path+'_2', use_irevnet=not options.use_resnet)
    train_set_3 = ZSLDataset(train_path+'_3', use_irevnet=not options.use_resnet)
    train_sets = [train_set_1, train_set_2, train_set_3]

    val_set_1 = ZSLDataset(validation_path+'_1', use_irevnet=not options.use_resnet)
    val_set_2 = ZSLDataset(validation_path+'_2', use_irevnet=not options.use_resnet)
    val_set_3 = ZSLDataset(validation_path+'_3', use_irevnet=not options.use_resnet)
    validation_sets = [val_set_1, val_set_2, val_set_3]

    for split in range(3):
        trainloader = DataLoader(train_sets[split],
                                 batch_size=options.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

        testloader = DataLoader(validation_sets[split],
                                batch_size=options.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=True)

        num_classes = train_sets[split].classes.shape[0]
        classes_enum = torch.tensor(np.array(range(num_classes), dtype=np.int64)).cuda()

        dim_semantic = train_sets[split][0]['class_embedding'].shape[0]
        dim_visual = train_sets[split][0]['image_embedding'].shape[0]
        dim_attributes = train_sets[split][0]['class_predicates'].shape[0]

        all_class_embeddings = torch.tensor(np.array(train_sets[split].class_embeddings)).float().cuda()
        all_class_predicates = torch.tensor(np.array(train_sets[split].class_predicates)).float().cuda()

        query_ids = set([validation_sets[split][i]['class_id'] for i in range(len(validation_sets[split]))])
        ids = list(i - 1 for i in query_ids)
        query_mask = np.zeros(num_classes)
        query_mask[ids] = 1
        query_mask = torch.tensor(query_mask, dtype=torch.int64).cuda()

        gamma = options.gamma

        alpha1 = options.alphas[0]  # triplet
        alpha2 = options.alphas[1]  # surjection
        alpha3 = options.alphas[2]  # l2 regularization

        margin = options.margin

        positive_part = torch.nn.ReLU().cuda()

        if options.only_words:
            v_to_s = ContinuousMap(dim_source=dim_visual, dim_dest=dim_semantic, width=512).cuda()
            s_to_v = ContinuousMap(dim_source=dim_semantic, dim_dest=dim_visual, width=512).cuda()

            if options.optimizer == 'adam':
                optimizer = torch.optim.Adam(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                             lr=options.learning_rate,
                                             betas=(0.9, 0.999),
                                             weight_decay=options.weight_decay)
            else:
                optimizer = torch.optim.SGD(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                            lr=options.learning_rate,
                                            momentum=options.momentum,
                                            weight_decay=options.weight_decay,
                                            nesterov=True)

            positive_part = torch.nn.ReLU().cuda()

            # Main Loop
            for e in range(options.n_epochs):
                v_to_s = v_to_s.train()
                s_to_v = s_to_v.train()

                running_loss = 0
                for i, sample in enumerate(trainloader):
                    optimizer.zero_grad()

                    batch_visual = sample['image_embedding'].float().cuda()
                    batch_classes = sample['class_id'].cuda() - 1

                    e_hat = v_to_s(s_to_v(all_class_embeddings))
                    delta = (e_hat - all_class_embeddings)
                    surjection_loss = (delta * delta).sum(dim=-1).mean()

                    s_out = v_to_s(batch_visual)

                    same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                    same_class = same_class.detach()

                    d_matrix = dist_matrix(s_out, all_class_embeddings)

                    closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                    furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                    l2_loss = (s_out * s_out).sum(dim=-1).mean()
                    loss = positive_part(furthest_positive - closest_negative + margin)
                    loss = alpha1 * loss.mean() + alpha2 * surjection_loss + alpha3 * l2_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                else:
                    print('Training Loss epoch {0}: {1}'.format(e + 1, running_loss / len(trainloader)))

                if (e + 1) % 50 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.7

                if (e + 1) % 5 == 0:
                    print('\n\n- Evaluation on epoch {}'.format(e + 1))

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

                            same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                            same_class = same_class.detach()

                            d_matrix = dist_matrix(s_out, all_class_embeddings)

                            c_hat = (d_matrix + (1 - query_mask).float() * 1e6).argmin(dim=-1)

                            closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                            furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                            loss = alpha1 * furthest_positive.mean()

                            avg_loss += loss.item()
                            avg_accuracy += (c_hat == batch_classes).float().mean().item()

                    avg_accuracy /= n
                    avg_loss /= n

                    print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))

        elif options.only_attributes:
            v_to_s = ContinuousMap(dim_source=dim_visual, dim_dest=dim_attributes, width=512).cuda()
            s_to_v = ContinuousMap(dim_source=dim_attributes, dim_dest=dim_visual, width=512).cuda()

            if options.optimizer == 'adam':
                optimizer = torch.optim.Adam(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                             lr=options.learning_rate,
                                             betas=(0.9, 0.999),
                                             weight_decay=options.weight_decay)
            else:
                optimizer = torch.optim.SGD(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                            lr=options.learning_rate,
                                            momentum=options.momentum,
                                            weight_decay=options.weight_decay,
                                            nesterov=True)

            positive_part = torch.nn.ReLU().cuda()

            # Main Loop
            for e in range(options.n_epochs):
                v_to_s = v_to_s.train()
                s_to_v = s_to_v.train()

                running_loss = 0
                for i, sample in enumerate(trainloader):
                    optimizer.zero_grad()

                    batch_visual = sample['image_embedding'].float().cuda()
                    batch_classes = sample['class_id'].cuda() - 1

                    e_hat = v_to_s(s_to_v(all_class_predicates))
                    delta = (e_hat - all_class_predicates)
                    surjection_loss = (delta * delta).sum(dim=-1).mean()

                    s_out = v_to_s(batch_visual)

                    same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                    same_class = same_class.detach()

                    d_matrix = dist_matrix(s_out, all_class_predicates)

                    closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                    furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                    l2_loss = (s_out * s_out).sum(dim=-1).mean()
                    loss = positive_part(furthest_positive - closest_negative + margin)
                    loss = alpha1 * loss.mean() + alpha2 * surjection_loss + alpha3 * l2_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                else:
                    print('Training Loss epoch {0}: {1}'.format(e + 1, running_loss / len(trainloader)))

                if (e + 1) % 50 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.7

                if (e + 1) % 5 == 0:
                    print('\n\n- Evaluation on epoch {}'.format(e + 1))

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

                            same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                            same_class = same_class.detach()

                            d_matrix = dist_matrix(s_out, all_class_predicates)

                            c_hat = (d_matrix + (1 - query_mask).float() * 1e6).argmin(dim=-1)

                            closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                            furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                            loss = alpha1 * furthest_positive.mean()

                            avg_loss += loss.item()
                            avg_accuracy += (c_hat == batch_classes).float().mean().item()

                    avg_accuracy /= n
                    avg_loss /= n

                    print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))

        else:
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
                                             weight_decay=options.weight_decay)
            else:
                optimizer = torch.optim.SGD(list(v_to_s.parameters()) + list(s_to_v.parameters()),
                                            lr=options.learning_rate,
                                            momentum=options.momentum,
                                            weight_decay=options.weight_decay,
                                            nesterov=True)

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

                    d_matrix = (1 - gamma) * dist_matrix(s_word, all_class_embeddings) + \
                               gamma * dist_matrix(s_attr, all_class_predicates)

                    closest_negative, _ = (d_matrix + same_class.float() * 1e6).min(dim=-1)
                    furthest_positive, _ = (d_matrix * same_class.float()).max(dim=-1)

                    l2_loss = (1 - gamma) * (s_word * s_word).sum(dim=-1).mean() + \
                              gamma * (s_attr * s_attr).sum(dim=-1).mean()
                    loss = positive_part(furthest_positive - closest_negative + margin)
                    loss = alpha1 * loss.mean() + alpha2 * surjection_loss + alpha3 * l2_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                else:
                    print('Training Loss epoch {0}: {1}'.format(e + 1, running_loss / len(trainloader)))

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

                    print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))

        print('Split {0} done.'.format(split+1))


if __name__ == '__main__':
    main()
