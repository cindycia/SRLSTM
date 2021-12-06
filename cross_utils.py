import ntpath
import os

CROSS_AGENT_TYPE_DICT = dict()


def translate_type(type: str):
    if type == "moped" or type == "scooter" or type == "motorbike" or type == "motorcycle" or type == "Scooter":
        return "Scooter"
    elif type == "pedestrian" or type == "people" or type == "People":
        return "People"
    elif type == "car" or type == "Car":
        return "Car"
    elif type == "van" or type == "Van":
        return "Van"
    elif type == "bus" or type == "Bus":
        return "Bus"
    elif type == "jeep" or type == "Jeep":
        return "Jeep"
    elif type == "bicycle" or type == "Bicycle":
        return "Bicycle"
    elif type == "electric_tricycle" or type == "Electric_Tricycle":
        return "Electric_Tricycle"
    elif type == "gyro_scooter" or type == "Gyro_Scooter":
        return "Gyro_Scooter"
    else:
        return "Car"


def load_cross_dataset(gamma_root='/home/panpan/workspace/gamma',
                       folder='dataset/Cross_ct/',
                       file_name_pattern="frame"):
    global CROSS_AGENT_TYPE_DICT
    # glob all files from the folder
    files = []
    for file in os.listdir(os.path.join(gamma_root, folder)):
        if file.startswith(file_name_pattern):
            files.append(os.path.join(os.path.join(gamma_root, folder), file))
    # print(files)

    # Read each file line by line
    for file in files:
        with open(file) as f:
            count = 0
            lines = f.readlines()
            for line in lines:
                count += 1
                data = line.split(' ')
                agent_id = int(data[0])
                frame_id = int(data[1])
                agent_type = data[-2]
                # print("frame {}, agent {}, type {}".format(frame_id, agent_id, agent_type))
                # key = 'tf_' + data[0] + '_a_' + data[1]
                key = data[0]
                if key in CROSS_AGENT_TYPE_DICT.keys():
                    if CROSS_AGENT_TYPE_DICT[key] != translate_type(agent_type):
                        print(f'agent {key} in frame {frame_id} type {CROSS_AGENT_TYPE_DICT[key]} '
                              f'mismatch with {translate_type(agent_type)}')
                else:
                    CROSS_AGENT_TYPE_DICT[key] = translate_type(agent_type)
    # print(f'type dict:\n {CROSS_AGENT_TYPE_DICT}')


def get_ped_type_from_cross(ped_key):
    return "People" # for debugging purpose
    return CROSS_AGENT_TYPE_DICT[ped_key]


if __name__ == '__main__':
    load_cross_dataset()
