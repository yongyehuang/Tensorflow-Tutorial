# -*- coding:utf-8 -*- 

import logging
import logging.handlers

"""the use of logging.
refer: http://blog.csdn.net/chosen0ne/article/details/7319306

- handler：将日志记录（log record）发送到合适的目的地（destination），比如文件，socket等。
    一个logger对象可以通过addHandler方法添加0到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示。
- formatter：指定日志记录输出的具体格式。formatter的构造方法需要两个参数：消息的格式字符串和日期字符串，这两个参数都是可选的。
"""

# 1.保存至日志文件
LOG_FILE = 'training.log'
# 2.设置日志，maxBytes设置文件大小，backpuCount设置文件数量
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
# 3.设置消息输出格式
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('train')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)

for i in range(10):
    logger.info('first info message %d' % i)
    logger.debug('first debug message %d' % i)
