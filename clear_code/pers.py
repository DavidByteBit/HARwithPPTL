import numpy as np


class personalizer:

    def __init__(self, label_space, feature_extractor):
        self.label_space = label_space
        self.feature_extractor = feature_extractor
        self.weight_matrix = None

    def classify(self, unlabled_data):

        assert(self.weight_matrix is not None)

        feature_extracted_instances = self.feature_extractor(unlabled_data)
        class_label_count = len(self.label_space)
        weight_matrix = self.weight_matrix
        num_of_instances = len(unlabled_data)

        raw_classification_matrix = np.empty((num_of_instances, class_label_count))

        for i in range(num_of_instances):
            projected_data = feature_extracted_instances[i]
            for j in range(class_label_count):
                raw_classification_matrix[i][j] = np.dot(weight_matrix[j], projected_data)

        # print(raw_classification_matrix)

        return np.argmax(raw_classification_matrix, axis=1)

        # similarity_vector = [[] for i in range(class_label_count)]
        # projected_data = feature_extracted_instances[0]
        #
        # for j in range(class_label_count):
        #     similarity_vector[j] = np.dot(weight_matrix[j], projected_data)
        #
        # if np.isnan(np.sum(similarity_vector)):
        #     print(similarity_vector)
        #
        # return np.argmax(similarity_vector)

    # This method has to be called before 'classify'
    def initialize_weight_matrix(self, settings_map, source_data, target_data):

        source_features = source_data[0]
        target_features = target_data[0]
        source_labels = source_data[1]
        target_labels = target_data[1]

        dataset = np.concatenate((source_features, target_features))
        labels = np.concatenate((source_labels, target_labels))

        feature_extractor = self.feature_extractor
        label_space = self.label_space

        dataset = feature_extractor(dataset)

        path_to_save_forward_result = settings_map["path_to_this_repo"] + "/storage/results/itc/forwad_pass.save"

        # save results so we can validate secure version later
        with open(path_to_save_forward_result, 'w') as stream:

            dataset_string = []

            for forward_res in dataset:
                dataset_string.append(str([['{:.7f}'.format(b) for b in a] for a in forward_res]))

            dataset_string = "".join(dataset_string)

            stream.write(dataset_string)

        # with open("correct_answer.txt", 'w') as f:
        #     for r in dataset:
        #         f.write(str(r))
        #         f.write("\n\n\n\n")

        class_label_count = len(label_space)

        subsets = [[] for i in range(class_label_count)]
        indices = np.array([k for k in range(class_label_count)])

        for j in range(class_label_count):
            for i in range(len(dataset)):
                label = labels[i]
                dp = np.dot(indices, label)
                if dp == j:  # if label is j
                    subsets[j].append(dataset[i])

        weight_matrix_mean = [[0 for j in range(len(subsets[0][0]))] for i in range(class_label_count)]

        weight_matrix = [[] for i in range(class_label_count)]

        for j in range(class_label_count):

            for row in subsets[j]:
                weight_matrix_mean[j] = np.add(weight_matrix_mean[j], row)

            weight_matrix[j] = np.divide(weight_matrix_mean[j], np.linalg.norm(weight_matrix_mean[j].astype(float)))

            # print(weight_matrix[j])

        weight_matrix_mean = np.array(weight_matrix_mean)
        weight_matrix = np.array(weight_matrix)

        self.weight_matrix = weight_matrix
