import learners.svm_learners.simple_svm_learner as simple_svm_learner

def online_svm(self, training_data):
    my_data = build_dataset_from_file(training_data)
    learner = simple_svm_learner.SimpleLearner(my_data)
    learner.label_instances(learner.pick_balanced_initial_training_set(1))
    learner.rebuild_models()
    learner.active_learn(100, batch_size=1)
