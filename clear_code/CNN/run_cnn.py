from . import nets

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import mean
from numpy import std


class train_CNN:

    def __init__(self, trainX, trainy, testX, testy):
        self.trainX = trainX
        self.trainy = trainy
        self.valX = testX
        self.valy = testy
        self.testX = testX
        self.testy = testy

        print("shape of training set: " + str(trainX.shape))
        print("shape of corresponding labels: " + str(trainy.shape))

        print("shape of testing set: " + str(testX.shape))
        print("shape of corresponding labels: " + str(testy.shape))


    def evaluate_model(self, settings_map, epochs=50, net=1):

        path_to_this_repo = settings_map["path_to_this_repo"]

        verbose, batch_size = None, None

        # if net == 1:
        #     verbose, batch_size = 0, 64
        # elif net == 2:
        #     verbose, batch_size = 0, 64

        verbose, batch_size = 0, 64

        checkpoint_path = settings_map["cnn_path"]
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        n_timesteps, n_features, n_outputs = self.trainX.shape[1], self.trainX.shape[2], self.trainy.shape[1]

        net_string = settings_map["net"]
        model = nets.nets().models[net_string](n_timesteps, n_features, n_outputs)

        # model = None
        # if net == 1:
        #     model = nets.nets().create_model1(n_timesteps, n_features, n_outputs)
        # elif net == 2:
        #     model = nets.nets().create_model2(n_timesteps, n_features, n_outputs)

        model.save_weights(checkpoint_path.format(epoch=0))

        # fit network
        history = model.fit(self.trainX, self.trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,
                            validation_data=(self.valX, self.valy), callbacks=cp_callback)

        # plot
        if len(self.valX) > 0:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(
                ['train', 'val'], loc='upper left')
            plt.savefig(path_to_this_repo + '/clear_code/CNN/plots/accuracycurve.png')

            plt.clf()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(path_to_this_repo + '/clear_code/CNN/plots/losscurve.png')

        # test on test data
        if len(self.testy) > 0:
            results = model.evaluate(self.testX, self.testy, batch_size=batch_size, verbose=0)
            accuracy = results[1]
            # print(results)
            return accuracy
        return -1

    # summarize scores
    def summarize_results(self, scores):
        print(scores)
        m, s = mean(scores), std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    # run an experiment
    def run_experiment(self, settings_map, repeats=10, epochs=50, net=1):
        # repeat experiment
        scores = list()

        print("running experiments")

        for r in range(repeats):
            score = self.evaluate_model(settings_map, epochs=epochs,net=net)
            score = score * 100.0
            print('>#%d: %.3f' % (r + 1, score))
            scores.append(score)
        # summarize results
        self.summarize_results(scores)
        return scores[0]

