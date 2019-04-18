import tensorflow as tf
import logistic_regression

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:128.105.144.12"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node0:128.105.144.12"
    ],
    "worker" : [
        "host_name0:128.105.144.12",
        "host_name1:128.105.144.28"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "host_name0:128.105.144.12"
    ],
    "worker" : [
        "host_name0:128.105.144.12",
        "host_name1:128.105.144.28",
        "host_name2:128.105.144.16",
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
    logistic_regression
    #put your code here

