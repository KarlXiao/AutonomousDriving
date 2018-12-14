import os
import sys
import numpy as np
import argparse
import tensorflow as tf
from core import PerceptionNet
from core import create_loader

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='checkpoint', help='directory to save checkpoint')
parser.add_argument('--log', type=str, default='log/', help='directory to save training log')
parser.add_argument('--resume_dir', type=str, default=None, help='model directory for finetune training')
parser.add_argument('--capacity', type=int, default=1000, help='maximum number of elements in the queue')
parser.add_argument('--train_data', type=str, default='data/val.tfrecords', help='tfrecords to load')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--epoch', type=int, default=1600, help='number of training epoch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--decay_step', default=10000, type=float, help='learning rate decay step')
parser.add_argument('--decay_rate', default=0.5, type=float, help='learning rate decay rate')


def grad(model, x, seg, box, label, number):
    with tf.GradientTape() as tape:
        predictions = model(x, True)
        loss = model.loss(predictions, seg, box, label, number)
    return tape.gradient(loss, model.variables), loss


def train(unparsed):

    best_loss = tf.convert_to_tensor(np.inf, tf.float32)

    train_dataset = create_loader(FLAGS.train_data, FLAGS.batch_size, FLAGS.capacity)

    model = PerceptionNet([1, 1, 1, 1])

    step_counter = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lr, step_counter, FLAGS.decay_step, FLAGS.decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)
    checkpoint_path = os.path.join(FLAGS.save_dir, 'ckpt')
    #######################################################################################

    if FLAGS.resume_dir:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS.resume_dir))

    for epoch in np.arange(FLAGS.epoch):

        average_loss = 0.0
        for idx, (images, seg, box, label, number) in enumerate(train_dataset):

            grads, loss = grad(model, images, seg, box, label, number)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

            average_loss += loss
            print('Batch size:{}, iter:{}, loss:{:.4f}'.format(FLAGS.batch_size, step_counter.numpy(), loss))

        average_loss /= (idx+1)

        print('==== Epoch:{}, Avg loss:{:.4f} ====\n'.format(epoch, average_loss))

        if (best_loss > average_loss):
            best_loss = average_loss
            checkpoint.save(checkpoint_path)
            print('Iter: {} model is saved'.format(step_counter.numpy()))


if __name__ == '__main__':

    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.log):
        os.makedirs(FLAGS.log)

    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
