
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
        total_hit_count = 0
        print
        for fold in range(folds):
            training_set = data.select(cv_indices, fold, negate=1)
            test_set = data.select(cv_indices, fold)

            model = learner(training_set)
            hit_count = 0
            for example in test_set:
                prediction = classifier(model, example)
                correct = example.getclass()
                if prediction == correct:
                    hit_count += 1
            print "%.2f %% accuracy (%d of %d examples)" % (
                float(hit_count) / len(test_set) * 100, hit_count, len(test_set))
            total_hit_count += hit_count
        print "%.2f %% average accuracy" % \
                (total_hit_count / float(folds) / len(test_set) * 100)


