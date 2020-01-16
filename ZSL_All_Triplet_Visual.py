import torch
import torch.cuda
import numpy as np

from torch.utils.data import DataLoader
from zsldataset import ZSLDataset
from models import EncoderAttributes, DecoderAttributes

from ZSL_Words_L2 import parse_args, dist_matrix


def dist_cos_matrix(batch1, batch2):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    dist_matrix = cos(batch1.unsqueeze(1), batch2.unsqueeze(0))
    return dist_matrix

def main():
    options = parse_args()

    # Load Data
    if options.leonhard:
        train_path = 'Data/CUB/trainvalclasses'
        test_path = 'Data/CUB/testclasses'
    else:
        train_path = 'Data/CUB/trainvalclasses'
        test_path = 'Data/CUB/testclasses'

    trainset = ZSLDataset(train_path, use_predicates=True, use_irevnet=options.use_irevnet)
    testset = ZSLDataset(test_path, use_predicates=True, use_irevnet=options.use_irevnet)

    num_classes = trainset.classes.shape[0]

    dim_semantic = trainset[0]['class_embedding'].shape[0]
    dim_visual = trainset[0]['image_embedding'].shape[0]
    dim_attributes = trainset[0]['class_predicates'].shape[0]

    all_class_embeddings = torch.tensor(np.array(trainset.class_embeddings)).float().cuda()
    all_class_predicates = torch.tensor(np.array(trainset.class_predicates)).float().cuda()
    classes_enum = torch.tensor(np.array(range(num_classes), dtype=np.int64)).cuda()
    all_train_image_embeddings = torch.tensor(np.array(trainset.image_embeddings)).cuda().float()
    all_train_labels = torch.tensor(trainset.labels['class_id'].values).cuda() - 1

    #Find median image embeddings for each class
    image_mean = all_train_image_embeddings.mean(0)
    all_train_image_embeddings = all_train_image_embeddings - image_mean
    mask = all_train_labels.unsqueeze(0) == classes_enum.unsqueeze(1)
    all_train_image_embeddings = torch.stack(
        [torch.median(all_train_image_embeddings[mask[i]], dim=0)[0] for i in set(all_train_labels.tolist())])

    all_train_labels = all_train_labels.unique() #having reduced all image embeddings of a class to a single embedding we remove all duplicate labels here

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
                                     weight_decay=1e-2)
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


    alpha1 = options.alphas[0]  # triplet
    alpha2 = options.alphas[1]  # surjection
    alpha3 = options.alphas[2]  # l2 regularization
    gamma = options.gamma
    margin = options.margin

    # Main Loop
    for e in range(options.n_epochs):
        v_to_s = v_to_s.train()
        s_to_v = s_to_v.train()

        running_loss = 0
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()


            batch_classes = sample['class_id'].cuda() - 1
            batch_semantic = sample['class_embedding'].cuda().float()
            batch_predicates = sample['class_predicates'].cuda().float()

            e_hat = v_to_s(s_to_v(all_class_embeddings, all_class_predicates))
            delta = (e_hat[1] - all_class_embeddings)
            surjection_loss = (delta * delta).sum(dim=-1).mean()
            delta = (e_hat[0] - all_class_predicates)
            surjection_loss = (1 - gamma) * surjection_loss + gamma * (delta * delta).sum(dim=-1).mean()

            ##Triplet loss in visual space
            same_class = all_train_labels.unsqueeze(0) == batch_classes.unsqueeze(1)
            same_class = same_class.detach()
            v_out = s_to_v(batch_semantic, batch_predicates)
            d_matrix_v = dist_cos_matrix(v_out, all_train_image_embeddings)

            closest_negative, _ = (d_matrix_v + same_class.float() * 1e6).min(dim=-1)

            furthest_positive, _ = (d_matrix_v * same_class.float()).max(dim=-1)
            l2_loss = (v_out * v_out).sum(dim=-1).mean()


            loss = positive_part(furthest_positive - closest_negative + margin)
            loss = alpha1 * loss.mean() + alpha2 * surjection_loss + alpha3 * l2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Training Loss epoch {0}: {1}'.format(e+1, running_loss/len(trainloader)))

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
            v_out = s_to_v(all_class_embeddings, all_class_predicates)
            with torch.no_grad():
                for i, sample in enumerate(testloader):
                    n += 1


                    batch_classes = sample['class_id'].cuda() - 1
                    batch_visual = sample['image_embedding'].cuda().float() - image_mean


                    ##Triplet loss in visual space
                    d_matrix_v = dist_cos_matrix(batch_visual, v_out)
                    same_class = classes_enum.unsqueeze(0) == batch_classes.unsqueeze(1)
                    same_class = same_class.detach()

                    closest_negative, _ = (d_matrix_v + same_class.float() * 1e6).min(dim=-1)
                    furthest_positive, _ = (d_matrix_v * same_class.float()).max(dim=-1)



                    loss = alpha1 * furthest_positive.mean()

                    avg_loss += loss.item()

                    #find nearest neighbour
                    c_hat = (d_matrix_v + (1 - query_mask).float() * 1e9).argmin(dim=-1)
                    avg_accuracy += (c_hat == batch_classes).float().mean().item()

            avg_accuracy /= n
            avg_loss /= n

            print('Average acc.: {}, Average loss:{}\n\n'.format(avg_accuracy, avg_loss))


if __name__ == '__main__':
    main()
