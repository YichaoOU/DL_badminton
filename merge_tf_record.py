import tensorflow as tf

def merge_tfrecords(input_files, output_file):
    """
    Merges multiple TFRecord files into a single TFRecord file.
    
    Parameters:
    input_files (list): List of paths to input TFRecord files.
    output_file (str): Path to the output TFRecord file.
    """
    # Create a TFRecord writer to write the combined data
    with tf.io.TFRecordWriter(output_file) as writer:
        # Iterate over all the input files
        for input_file in input_files:
            # Read the input TFRecord file
            raw_dataset = tf.data.TFRecordDataset(input_file)
            
            # Iterate through each example in the dataset
            for raw_record in raw_dataset:
                # Write each record to the output file
                writer.write(raw_record.numpy())

    print(f"TFRecords merged into {output_file}")


import glob

input_files=glob.glob("*test.record")+glob.glob("*/*test.record")
output_file = 'merged_test.tfrecord'
merge_tfrecords(input_files, output_file)

input_files=glob.glob("*train.record")+glob.glob("*/*train.record")
output_file = 'merged_train.tfrecord'
merge_tfrecords(input_files, output_file)