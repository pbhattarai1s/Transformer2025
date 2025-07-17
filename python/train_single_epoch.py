import torch

# this is a function to run one single epoch 
def run_epoch(model, dataloader, model_params, criterion, criterion_adv, experiment, epoch, 
              is_train=True, optimizer=None, max_batches=None):

    total_class_loss = 0.0
    total_advers_loss = 0.0
    total_tot_loss = 0.0
    step_losses_classifier = []
    step_losses_advers = []
    step_losses_tot = []

    for batch_idx, (event_feats, object_feats, labels, valid_mask, mbb) in enumerate(dataloader):
        if is_train:
            optimizer.zero_grad()

        mbb_bins = torch.bucketize(mbb / 1000.0, torch.tensor(model_params['mbb_bins'], device=mbb.device)) - 1
        class_outputs, adv_output = model(event_feats, object_feats, valid_mask)
        loss = criterion(class_outputs.squeeze(), labels.float())

        step_losses_classifier.append(loss.item())
        experiment.log_metric(
            f"{'train' if is_train else 'val'}_batch_classifier_loss", 
            loss.item(), 
            step=epoch * len(dataloader) + batch_idx
        )
        total_class_loss += loss.item()

        if adv_output is not None:
            loss_adv = criterion_adv(adv_output, mbb_bins)
            loss += model_params['adv_lambda'] * loss_adv

            total_advers_loss += loss_adv.item()
            total_tot_loss += loss.item()
            step_losses_advers.append(loss_adv.item())
            step_losses_tot.append(loss.item())

            experiment.log_metric(
                f"{'train' if is_train else 'val'}_batch_advers_loss", 
                loss_adv.item(), 
                step=epoch * len(dataloader) + batch_idx
            )
            experiment.log_metric(
                f"{'train' if is_train else 'val'}_batch_tot_loss", 
                loss.item(), 
                step=epoch * len(dataloader) + batch_idx
            )

        if is_train:
            loss.backward()
            optimizer.step()

        if max_batches is not None and batch_idx >= max_batches:
            break

    num_batches = batch_idx + 1
    avg_class_loss = total_class_loss / num_batches
    avg_advers_loss = total_advers_loss / num_batches if total_advers_loss > 0 else 0.0
    avg_tot_loss = total_tot_loss / num_batches if total_tot_loss > 0 else avg_class_loss

    return avg_class_loss, avg_advers_loss, avg_tot_loss, step_losses_classifier, step_losses_advers, step_losses_tot
