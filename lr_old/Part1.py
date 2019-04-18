import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import time

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
tf.app.flags.DEFINE_integer("sync", 0, "Wether use Distributed Synchronization Mode to update parameter")
FLAGS = tf.app.flags.FLAGS
sync = FLAGS.sync
#parameters
learning_rate = 0.01
total_epochs = 25
print_epoch_step = 5
batch_size = 100


tf.logging.set_verbosity(tf.logging.DEBUG)
#TODO: change host names
clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "node0:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222",
        "node2:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    #put your code here
    with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo
        )):
    #initialize tf variables and inputs
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        
        dataset = input_data.read_data_sets("/tmp/data/", one_hot=True)
        #initialize operations
        prediction  = tf.nn.softmax(tf.matmul(x, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # if FLAGS.deploy_mode == "single":
        #     train_operation = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        #     with tf.Session() as sess:
        #         sess.run(init)
        #         for epo in range(total_epochs):
        #             batch_number = math.floor(dataset.train.num_examples/batch_size)

        #             for bat in range(batch_number):
        #                 tmpx, tmpy = dataset.train.next_batch(batch_size)
        #                 _, L = sess.run([train_operation, loss], feed_dict={x:tmpx, y:tmpy})

        #                 if(epoch+1) % print_epoch_step == 0:
        #                     print("Epoch:", '%04d' % (epoch+1), "loss:", "{:.9f}".format(L))

        #         #hits = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #         #accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))
        #         #print("accuracy:", accuracy.eval({x: minst.test.images, y: minst.test.labels}))
        # else: 

        grads_and_vars = optimizer.compute_gradients(loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        init = tf.global_variables_initializer()
        if FLAGS.deploy_mode == "single":
            worker_number = 1
        elif FLAGS.deploy_mode == "cluster":
            worker_number = 2
        elif FLAGS.deploy_mode == "cluster2":
            worker_number = 3

        if sync == 1:
            replica_optimizer = tf.train.SyncReplicasOptimizer(optimizer, 
                                                    replicas_to_aggregate=worker_number,
                                                    #replica_id=FLAGS.task_index,
                                                    total_num_replicas=worker_number,
                                                    #use_locking=True
                                        )
            train_operation = replica_optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step
                                                )

            init_tokens_operation = replica_optimizer.get_init_tokens_op()
            queue_runner = replica_optimizer.get_chief_queue_runner()
        else:
            train_operation = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step
                                                )
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                                         #logdir="./checkpoint/",
                                 init_op=init,
                                 summary_op=None,
                                 global_step=global_step,
                                         #save_model_secs=60                                        
                                )
        with sv.prepare_or_wait_for_session(server.target) as sess:
            starttime = time.clock()
            if FLAGS.task_index == 0 and sync == 1:
                sv.start_queue_runners(sess, [queue_runner])
                sess.run(init_tokens_operation)
            for epo in range(total_epochs):
                batch_number = int(dataset.train.num_examples/batch_size)

                for bat in range(batch_number):
                    tmpx, tmpy = dataset.train.next_batch(batch_size)
                    _, L = sess.run([train_operation, loss], feed_dict={x:tmpx, y:tmpy})

                print ("Epoch time:" , time.clock() - starttime)
                starttime = time.clock()
                if(epo+1) % print_epoch_step == 0 and FLAGS.task_index == 0:
                    print("Epoch:", '%02d' % (epo+1), "loss:", "{:.5f}".format(L))
        #sv.stop()

