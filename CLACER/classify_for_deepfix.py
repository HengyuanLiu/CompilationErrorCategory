import time
import sqlite3
import SyntaticErrorClassifier

print("-"*15 + "| start log at " + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + " |" + "-"*15)

db_path = r'E:\PYprojects\DeepFixHelper\data\iitk-dataset\dataset_train.db'
# db_path = r'E:\PYprojects\DeepFixHelper\data\iitk-dataset\dataset_train_id.db'

# # 增加数据库里面的列
# with sqlite3.connect(db_path) as conn:
#     conn.execute('''ALTER TABLE train_data ADD error_class_id integer;''')

tuples = []
wrong_id = []
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    for row in cursor.execute("SELECT id, detokenized_code FROM train_data;"):
        code_id = row[0]
        code = row[1]
        try:
            class_id = SyntaticErrorClassifier.syntactic_error_classifier(code)
        except Exception as e:
            wrong_id.append(code_id)
            class_id = -1

        tuples.append((class_id, code_id))

        print(code_id)

with open('wrong_code_id.txt', mode='w') as fp:
    fp.write(str(wrong_id))

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE train_data SET error_class_id=? WHERE id=?;", tuples)
    conn.commit()

print("-"*15 + "| end log at " + time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())) + " |" + "-"*15)
