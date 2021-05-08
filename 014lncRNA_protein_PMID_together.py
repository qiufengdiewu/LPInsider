# -*- coding: utf-8 -*-
import MySQLdb as mySQLDB

def saveData():

    # 打开数据库连接
    db = mySQLDB.connect(host='127.0.0.1', user='root', passwd='11223366', db='ppi_corpus', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL查询语句
    sql = "select pmid from lncrna_pmid "  # 获取PID的记录条数总数
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        lncRNA_PMID_list = cursor.fetchall()
    except:
        print("Error")

    print (len(lncRNA_PMID_list))

    for i in range(len(lncRNA_PMID_list)):
        lncRNA_PMID=int(lncRNA_PMID_list[i][0])
        sql = "select pmid from protein_pmid where pmid="+str(lncRNA_PMID)+";"
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            common_pmid = cursor.fetchall()
        except:
            print("Error")

        if len(common_pmid)>0:
            sql = "insert into lncrna_protein_together (pmid,count) values(" + str(common_pmid[0][0]) + ",1);"
            try:
                # 执行SQL语句
                cursor.execute(sql)
                # 提交到数据库执行
                db.commit()
            except:
                # Rollback in case there is any error
                db.rollback()
                print("Error,can't insert data.  " + str(sql))

    db.close()


if __name__ == "__main__":
    saveData()
