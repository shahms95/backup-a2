import tensorflow as tf
import time

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
tf.app.flags.DEFINE_integer("node_num", 0, "number of nodes")
tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")
FLAGS = tf.app.flags.FLAGS
node_num = FLAGS.node_num
tf.logging.set_verbosity(tf.logging.DEBUG)



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
issync = FLAGS.issync

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # logistic_regression
    #put your code here
    
    with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster = clusterinfo
        )):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # Parameters
        learning_rate = 0.01
        training_epochs = 25
        batch_size = 100
        display_step = 5


        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

        if issync == 0:

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cost)

            train_op = optimizer.apply_gradients(grads_and_vars,
                                                      global_step=global_step 
                                                     )
            init = tf.global_variables_initializer()
            last_time = time.clock()               
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                                     init_op=init,
                                     summary_op=None, #summary_op,                                
                                     global_step=global_step                              
                                    )                
            with sv.prepare_or_wait_for_session(server.target) as sess:
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                    for i in range(total_batch):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([train_op, cost], feed_dict={x: batch_xs,
                                                                      y: batch_ys})
                        # Compute average loss
                        avg_cost += c / total_batch
                    # Display logs per epoch step
                    if (epoch+1) % display_step == 0 and FLAGS.task_index == 0:
                        print time.clock() - last_time
                        last_time = time.clock()
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # sv.stop()
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cost)

            # if issync == 1:
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=3,
                                   total_num_replicas=3)

       

            train_op = optimizer.apply_gradients(grads_and_vars,
                                                      global_step=global_step 
                                                     )

            init_token_op = optimizer.get_init_tokens_op()
            chief_queue_runner = optimizer.get_chief_queue_runner()

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()

            last_time = time.clock()

                
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                                     init_op=init,
                                     summary_op=None, #summary_op,                                
                                     global_step=global_step                              
                                    )                

            with sv.prepare_or_wait_for_session(server.target) as sess:
                # If is Synchronization Mode
                # if FLAGS.task_index == 0 and issync == 1:
                if FLAGS.task_index == 0:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                    for i in range(total_batch):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([train_op, cost], feed_dict={x: batch_xs,
                                                                      y: batch_ys})
                        # Compute average loss
                        avg_cost += c / total_batch
                    # Display logs per epoch step
                    if (epoch+1) % display_step == 0:
                    #     print 1
                        print time.clock() - last_time
                        last_time = time.clock()
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # sv.stop()

        # Test model
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


