
def open_setting(source_file_org, source_file_new,
                 num_known, target_file_org=None, target_file_new=None,
                 validation_file_org=None, validation_file_new=None):

    # keep known source samples only
    for line in source_file_org:
        if int(line.strip().split(' ')[2]) < num_known:
            source_file_new.write(line)

    for line in target_file_org:
        if int(line.strip().split(' ')[2]) >= num_known:
            target_file_new.write(
                line.strip().split(' ')[0] +
                " " +
                line.strip().split(' ')[1] +
                " " +
                str(num_known) +
                "\n"
            )
        else:
            target_file_new.write(line)

    for line in validation_file_org:
        if int(line.strip().split(' ')[2]) >= num_known:
            validation_file_new.write(
                line.strip().split(' ')[0] +
                " " +
                line.strip().split(' ')[1] +
                " " +
                str(num_known) +
                "\n"
            )
        else:
            validation_file_new.write(line)

# e.g. Olympic -> UCF
source_file_org = open("dataset/olympic/" \
                  "list_olympic_train_ucf_olympic-feature_org.txt", "r")
source_file_new = open("dataset/olympic/" \
                  "list_olympic_train_ucf_olympic-feature.txt", "w+")
target_file_org = open("dataset/ucf101/" \
                  "list_ucf101_train_ucf_olympic-feature_org.txt", "r")
target_file_new = open("dataset/ucf101/" \
                  "list_ucf101_train_ucf_olympic-feature.txt", "w+")
validation_file_org = open("dataset/ucf101/" \
                  "list_ucf101_val_ucf_olympic-feature_org.txt", "r")
validation_file_new = open("dataset/ucf101/" \
                  "list_ucf101_val_ucf_olympic-feature.txt", "w+")

open_setting(source_file_org, source_file_new, 3, target_file_org=target_file_org, target_file_new=target_file_new, validation_file_org=validation_file_org, validation_file_new=validation_file_new)
