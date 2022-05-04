import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

def batch_loss_plot(filename, model, n_epochs, batch_size, task, gpu):

    with open(filename, 'r') as f:
        batch_loss = json.load(f)

    colors = ['r', 'b', 'g', 'o']

    n_batches= int(len(batch_loss)/n_epochs)
    plt.plot(range(1,n_batches+1), batch_loss[:n_batches],'r', label='Epoch 1')
    plt.plot(range(1, n_batches + 1), batch_loss[n_batches:2*n_batches], 'b', label='Epoch 2')
    plt.plot(range(1, n_batches + 1), batch_loss[2*n_batches:], 'g', label='Epoch 3')
    plt.title('Batch Loss (Model:{}, Task:{}, Batch Size:{} GPU:{})'.format(model,task,batch_size,gpu))
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Model:{}_Task:{}_Batch Size:{}_GPU:{}'.format(model,task,batch_size,gpu))
    plt.show()

def loss_plots(n_epochs, gpu):

    bert_a100 = pd.read_csv('a100_cola/bert_cola_train')
    roberta_a100 = pd.read_csv('a100_cola/roberta_cola_train')
    bert_train_loss_a100 = list(bert_a100.train_loss)
    bert_val_loss_a100 = list(bert_a100.val_loss)
    robert_train_loss_a100 = list(roberta_a100.train_loss)
    roberta_val_loss_a100 = list(roberta_a100.val_loss)


    plt.plot( range(1,n_epochs+1), bert_train_loss_a100,
              marker='o', linestyle='dashed',
              label='Bert train (A100)')
    plt.plot(range(1, n_epochs + 1), bert_val_loss_a100,
             marker='o', linestyle='dashed',
             label='Bert val (A100)')
    plt.plot(range(1, n_epochs + 1), robert_train_loss_a100,
             marker='o', linestyle='dashed',
             label='Roberta val (A100)')
    plt.plot(range(1, n_epochs + 1), roberta_val_loss_a100,
             marker='o', linestyle='dashed',
             label='Roerta val (A100)')

    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Bert-Roberta Loss Comparison')
    plt.legend()
    plt.savefig('bert_roberta_loss_a100')
    plt.show()
    return

def val_acc_plots(n_epochs, gpu):

    bert_a100 = pd.read_csv('a100_cola/bert_cola_train')
    roberta_a100 = pd.read_csv('a100_cola/roberta_cola_train')
    bert_val_acc_a100 = list(bert_a100.val_accuracy)
    roberta_val_acc_a100 = list(roberta_a100.val_accuracy)



    plt.plot( range(1,n_epochs+1), bert_val_acc_a100,
              marker='o', linestyle='dashed',
              label='Bert (A100)')
    plt.plot(range(1, n_epochs + 1), roberta_val_acc_a100,
             marker='o', linestyle='dashed',
             label='Roberta (A100)')

    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.title('Bert-Roberta Validation Accuracy')
    plt.legend()
    plt.savefig('bert_roberta_val_acc_a100')
    plt.show()
    return

def time_plots(n_epochs, gpu):

    bert_a100 = pd.read_csv('a100_cola/bert_cola_train')
    roberta_a100 = pd.read_csv('a100_cola/roberta_cola_train')
    bert_train_a100 = list(bert_a100.training_time)
    bert_val_a100 = list(bert_a100.val_time)
    robert_train_a100 = list(roberta_a100.training_time)
    roberta_val_a100 = list(roberta_a100.val_time)


    plt.plot( range(1,n_epochs+1), bert_train_a100,
              marker='o', linestyle='dashed',
              label='Bert train (A100)')
    plt.plot(range(1, n_epochs + 1), bert_val_a100,
             marker='o', linestyle='dashed',
             label='Bert val (A100)')
    plt.plot(range(1, n_epochs + 1), robert_train_a100,
             marker='o', linestyle='dashed',
             label='Roberta train (A100)')
    plt.plot(range(1, n_epochs + 1), roberta_val_a100,
             marker='o', linestyle='dashed',
             label='Roberta val (A100)')

    plt.xlabel('Epoch #')
    plt.ylabel('Time')
    plt.legend()
    plt.title('Bert-Roberta Time Comparison')
    plt.savefig('bert_roberta_time_a100')
    plt.show()
    return

def total_train_time_plot():

    with open('v100_cola/bert_train_time', 'r') as f:
        bert_time = f.read()
    bert_time = float(bert_time.split(' ')[1][:-1])

    with open('v100_cola/roberta_train_time', 'r') as f:
        roberta_time = f.read()
    roberta_time = float(roberta_time.split(' ')[1][:-1])

    with open('a100_cola/bert_train_time', 'r') as f:
        bert_time_2 = f.read()
    bert_time_2 = float(bert_time_2.split(' ')[1][:-1])

    with open('a100_cola/roberta_train_time', 'r') as f:
        roberta_time_2 = f.read()
    roberta_time_2 = float(roberta_time_2.split(' ')[1][:-1])

    plt.bar(['bert (V100)', 'roberta (V100)', 'bert (A100)', 'roberta (A100)'],
            [bert_time, roberta_time, bert_time_2, roberta_time_2])
    plt.ylim(40, None)
    plt.xlabel('Model and GPU')
    plt.ylabel('Training Time')
    plt.title('Bert-Roberta Traing Time Comparison')
    plt.savefig('bert_roberta_train_time')
    plt.show()
    return

batch_loss_plot('a100_cola/bert_batch_loss', 'bert', 3, 32, 'cola', 'A100')
batch_loss_plot('a100_cola/roberta_batch_loss', 'roberta', 3, 32, 'cola', 'A100')
loss_plots(3, 'A100')
val_acc_plots(3, 'A100')
time_plots(3, 'A100')
# total_train_time_plot()

