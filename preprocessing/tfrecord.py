import tensorflow as tf


class SentenceTFRecord():
    def __init__(self, dataset, output_path):
        self.dataset = dataset
        self.output_path = output_path

    def parse_sentences(self):
        writer = tf.python_io.TFRecordWriter(self.output_path)

        all_data, all_labels, all_sizes = self.dataset

        for data, labels, size in zip(all_data, all_labels, all_sizes):
            example = self.make_example(data, labels, size)
            writer.write(example.SerializeToString())

        writer.close()

    def make_example(self, data, labels, size):
        example = tf.train.SequenceExample()

        example.context.feature['size'].int64_list.value.append(size)

        sentence_tokens = example.feature_lists.feature_list['tokens']
        labels_tokens = example.feature_lists.feature_list['labels']

        for (token, label) in zip(data, labels):
            sentence_tokens.feature.add().int64_list.value.append(int(token))
            labels_tokens.feature.add().int64_list.value.append(int(label))

        return example
