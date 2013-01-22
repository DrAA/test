
    def train_bayes(self):
        data = orange.ExampleTable(self.options.training_data)
        learner = lambda training_set: orange.BayesLearner(training_set)
        bayes = learner(data)
        attributes = [a.name.upper() for a in data.domain][:-1]
        for index, attribute in enumerate(attributes):
            self.debugout("\n%s:" % attribute)
            for value, probs in bayes.conditionalDistributions[index].items():
                args = [value] + map(lambda p: probs[p], xrange(3))
                self.debugout("%s\t%5.4f\t%5.4f\t%5.4f" % tuple(args))

        self.cross_validation(data, learner)

    def cross_validation(self, data, learner, classifier=None, folds=8):
        classifier = classifier or (lambda model, example: model(example))
        cv_indices = orange.MakeRandomIndicesCV(data, folds)

