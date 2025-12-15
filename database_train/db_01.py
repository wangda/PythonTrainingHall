import os
import sqlite3
import datetime

# 1. 检查test.db是否存在，如果不存在则创建
# 使用绝对路径确保数据库文件创建在database_train目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
db_file = os.path.join(current_dir, 'test.db')
db_exists = os.path.exists(db_file)

# 2. 连接数据库
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# 3. 创建表user，字段为: id, name, gender, birth, address
# 如果数据库文件不存在，则创建表
if not db_exists:
    cursor.execute('''
    CREATE TABLE user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        gender TEXT,
        birth TEXT,
        address TEXT
    )
    ''')
    print("表 user 创建成功")
else:
    print("数据库已存在，跳过表创建步骤")

# 4. 插入10条数据
def insert_sample_data():
    # 检查表是否为空
    cursor.execute('SELECT COUNT(*) FROM user')
    count = cursor.fetchone()[0]
    
    if count == 0:
        # 准备10条样本数据
        sample_data = [
            ('张三', '男', '1990-01-01', '北京市海淀区'),
            ('李四', '男', '1992-03-15', '上海市浦东新区'),
            ('王五', '男', '1988-07-22', '广州市天河区'),
            ('赵六', '男', '1995-11-08', '深圳市南山区'),
            ('钱七', '女', '1991-05-30', '成都市武侯区'),
            ('孙八', '女', '1993-09-17', '杭州市西湖区'),
            ('周九', '男', '1987-12-25', '南京市鼓楼区'),
            ('吴十', '女', '1994-02-14', '武汉市洪山区'),
            ('郑十一', '男', '1989-06-06', '西安市雁塔区'),
            ('zhangshan', '男', '1997-08-18', '苏州市姑苏区')
        ]
        
        # 执行插入操作
        cursor.executemany('''
        INSERT INTO user (name, gender, birth, address)
        VALUES (?, ?, ?, ?)
        ''', sample_data)
        
        # 提交事务
        conn.commit()
        print(f"成功插入 {len(sample_data)} 条数据")
    else:
        print(f"表中已有 {count} 条数据，跳过插入操作")

# 5. 更新数据
def update_data():
    # 将李四的地址更新为"北京市朝阳区"
    cursor.execute('''
    UPDATE user SET address = ? WHERE name = ?
    ''', ('北京市朝阳区', '李四'))
    
    # 提交事务
    conn.commit()
    print(f"成功更新 {cursor.rowcount} 条数据")

# 6. 删除name=zhangshan的数据
def delete_data():
    # 删除name为zhangshan的数据
    cursor.execute('''
    DELETE FROM user WHERE name = ?
    ''', ('zhangshan',))
    
    # 提交事务
    conn.commit()
    print(f"成功删除 {cursor.rowcount} 条数据")

# 显示所有数据
def show_all_data():
    cursor.execute('SELECT * FROM user')
    rows = cursor.fetchall()
    
    print("\n当前user表中的所有数据:")
    print("ID\t姓名\t性别\t出生日期\t\t地址")
    print("-" * 70)
    
    for row in rows:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t\t{row[4]}")

# 执行操作
if __name__ == "__main__":
    # 插入样本数据
    insert_sample_data()
    
    # 显示插入后的数据
    show_all_data()
    
    # 更新数据
    update_data()
    
    # 删除数据
    delete_data()
    
    # 显示最终数据
    show_all_data()
    
    # 关闭连接
    cursor.close()
    conn.close()
    print("\n数据库连接已关闭")