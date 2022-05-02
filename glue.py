import os
import random

import numpy as np
import pandas as pd
import torch
import wget
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import (BertModel,
                          BertTokenizer,
                          RobertaModel,
                          RobertaTokenizer,
                          AdamW,
                          BertForSequenceClassification,
                          RobertaForSequenceClassification,
                          )
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef

from zipfile import ZipFile


class Glue:

    def __init__(self, model_name, task_name, n_labels):
        self.epochs = 3
        self.batch_size = 32
        self.task_name = task_name
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.model, self.tokenizer = self._get_model_and_tokenizer(model_name,
                                                                   #
                                                                   )
        self.model.to(self.device)

    def _get_model_and_tokenizer(self, model_name, n_labels=2,
                                 output_attentions=False, output_hidden_states=False):

        if model_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                  # n_labels,
                                                                  # output_attentions,
                                                                  # output_hidden_states,
                                                                  )
        elif model_name == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
            model = RobertaForSequenceClassification.from_pretrained('roberta-base')

        return model, tokenizer

    def get_glue_tasks(self):

        if self.task_name == 'cola':
            url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
            if not os.path.exists('GLUE/cola_public_1.1.zip'):
                wget.download(url, 'GLUE/cola_public_1.1.zip')
                with ZipFile('GLUE/cola_public_1.1.zip', 'r') as zipObj:
                    zipObj.extractall()
            # Training Data
            df = pd.read_csv("cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
                             names=['sentence_source', 'label', 'label_notes', 'sentence'])
            print('Number of training sentences: {:,}\n'.format(df.shape[0]))
            train_sentences = df.sentence.values
            train_labels = df.label.values

            # Test data
            df = pd.read_csv("cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None,
                             names=['sentence_source', 'label', 'label_notes', 'sentence'])
            print('Number of test sentences: {:,}\n'.format(df.shape[0]))
            test_sentences = df.sentence.values
            test_labels = df.label.values

        return (train_sentences, train_labels), (test_sentences, test_labels)

    def tokenize_data(self, sentences, labels):

        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=104,  # Pad & truncate all sentences.
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, train_dataloader, val_dataloader):
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        epoch_training_time, epoch_val_time = [], []
        epoch_train_accuracy, epoch_val_accuracy = [], []
        epoch_train_loss, epoch_val_loss = [], []
        epoch_data = {
            'train_loss': epoch_train_loss,
            'training_time': epoch_training_time,
            'val_loss': epoch_val_loss,
            'val_time': epoch_val_time,
            'val_accuracy': epoch_val_accuracy
        }

        start_time = time.time()

        for epoch_i in range(0, self.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = time.time() - t0
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()
                output = self.model(input_ids=b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels
                                    )
                loss = output.loss
                logits = output.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = time.time() - t0
            epoch_train_loss.append(avg_train_loss)
            epoch_training_time.append(training_time)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            if val_dataloader:
                print("")
                print("Running Validation...")

                val_loss, val_accuracy, val_time = self.val_test(val_dataloader)
                epoch_val_loss.append(val_loss)
                epoch_val_accuracy.append(val_accuracy)
                epoch_val_time.append(val_time)

        print("")
        print("Training complete!")
        total_training_time = time.time() - start_time

        return epoch_data, total_training_time

    def val_test(self, val_dataloader):

        t0 = time.time()
        self.model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in val_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                output = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                loss = output.loss
                logits = output.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        avg_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_loss = total_eval_loss / len(val_dataloader)
        test_time = time.time() - t0

        print("  Accuracy: {0:.2f}".format(avg_accuracy))

        return avg_loss, avg_accuracy, test_time

    def test(self, test_loader):

        self.model.eval()
        predictions, true_labels = [], []

        for batch in test_loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

        return predictions, true_labels

    def task_accuracy(self, predictions, true_labels):

        if self.task_name == 'cola':
            matthews_set = []
            print('Accuracy on the CoLA benchmark is measured using the Matthews correlation coefficient(MCC).')
            print('Calculating Matthews Corr. Coef. for each batch...')
            for i in range(len(true_labels)):
                # The predictions for this batch are a 2-column ndarray (one column for "0"
                # and one column for "1"). Pick the label with the highest value and turn this
                # in to a list of 0s and 1s.
                pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

                # Calculate and store the coef for this batch.
                matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
                matthews_set.append(matthews)

                # Combine the results across all batches.
                flat_predictions = np.concatenate(predictions, axis=0)
                # For each sample, pick the label (0 or 1) with the higher score.
                flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
                # Combine the correct labels for each batch into a single list.
                flat_true_labels = np.concatenate(true_labels, axis=0)
                # Calculate the MCC
                mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

                print('Total MCC: %.3f' % mcc)

                return mcc * 100

    def save_data(self, data, train=True):

        if train:
            if type(data) == dict:
                df = pd.DataFrame(data)
                df.to_csv(self.model_name + '_' + self.task_name + '_train')
            else:
                with open(self.model_name + '_train_time', 'a') as f:
                    f.write([self.task_name, data])
        else:
            with open(self.model_name + '_test_acc', 'a') as f:
                f.write([self.task_name, data])

    def run(self):

        train_data, test_data = self.get_glue_tasks()

        sentences, labels = train_data
        input_ids, attention_masks, labels = self.tokenize_data(sentences, labels)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=self.batch_size  # Trains with this batch size.
        )
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )

        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,
                               eps=1e-8
                               )
        total_steps = len(train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=total_steps)
        train_data, train_time = self.train(train_dataloader, validation_dataloader)
        self.save_data(train_data, True)
        self.save_data(train_time, True)

        # Testing Data
        sentences, labels = test_data
        input_ids, attention_masks, labels = self.tokenize_data(sentences, labels)
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data,
                                           sampler=prediction_sampler,
                                           batch_size=self.batch_size)
        print('Predicting labels for {:,} test sentences...'.format(len(labels)))
        predictions, true_labels = self.test(prediction_dataloader)
        accuracy = self.task_accuracy(predictions, true_labels)
        self.save_data(accuracy, False)


if __name__ == '__main__':
    obj = Glue('roberta', 'cola', 2)
    obj.run()
