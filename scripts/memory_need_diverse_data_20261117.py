"""Hand-written memory-need routing examples, split by semantic template.

need=1: fulfilling the request requires information from prior personal experience.
need=0: the request is answerable from the prompt or general knowledge, or does
not ask for retrieval. Labels assume no additional conversation is supplied.
These small, constructed splits are not a representative population benchmark.
"""

TRAIN = [
    # Prior personal experience
    {'text': '按照我以前反馈过的过敏情况，推荐午餐时需要避开哪些食材？', 'need': 1},
    {'text': '回顾咱们上次讨论的搬家计划，先处理哪件事？', 'need': 1},
    {'text': '我之前提到过的项目卡点是什么，请帮我归纳一下。', 'need': 1},
    {'text': '结合我过去的跑步训练情况，建议这周的训练量。', 'need': 1},
    {'text': '按照我曾经说过的沟通偏好，帮我草拟一封邮件。', 'need': 1},
    {'text': '我报名那个课程时遇到过什么困难？', 'need': 1},
    # General knowledge
    {'text': '彩虹通常有哪些颜色？', 'need': 0},
    {'text': '地球绕什么天体公转？', 'need': 0},
    {'text': '解释一下什么是蒸发。', 'need': 0},
    {'text': '三角形的内角和是多少度？', 'need': 0},
    {'text': '大象属于哺乳动物吗？', 'need': 0},
    {'text': '汉字“森”由几个木字组成？', 'need': 0},
    # Prompt-contained information
    {'text': '箱子里有三支铅笔和两块橡皮，一共有几件物品？', 'need': 0},
    {'text': '会议定在周四下午三点，请提取会议时间。', 'need': 0},
    {'text': '阿岚把文件放进蓝色抽屉，文件在哪个抽屉？', 'need': 0},
    {'text': '原价八十元，打八折后是多少元？', 'need': 0},
    {'text': '便条写着“先浇花，再关窗”，第一步是什么？', 'need': 0},
    {'text': '这段话中提到的交通工具是：她骑自行车去了图书馆。', 'need': 0},
    # Non-retrieval requests
    {'text': '请把这句话改得更礼貌：明天把材料交来。', 'need': 0},
    {'text': '帮我拟一个春游活动标题。', 'need': 0},
    {'text': '请保持安静，不要查找我的任何资料。', 'need': 0},
    {'text': '把“今天阳光很好”翻译成英文。', 'need': 0},
    {'text': '为一只虚构的小狐狸编一首短诗。', 'need': 0},
    {'text': '列出整理书桌的三个通用步骤。', 'need': 0},
]

CAL = [
    # Prior personal experience
    {'text': '基于我过去描述的工作安排，找一个适合我锻炼的时段。', 'need': 1},
    {'text': '我曾经向你抱怨过哪种噪声？', 'need': 1},
    {'text': '延续我们以前为家庭预算定下的目标，给出下一步建议。', 'need': 1},
    {'text': '从我先前分享的旅行经历中，挑一件最让我自豪的事。', 'need': 1},
    # General knowledge
    {'text': '指南针的指针通常指向哪个方向？', 'need': 0},
    {'text': '植物通过什么过程利用阳光制造养分？', 'need': 0},
    {'text': '公历闰年二月有多少天？', 'need': 0},
    {'text': '解释海水涨落的潮汐现象。', 'need': 0},
    # Prompt-contained information
    {'text': '报名表写明名额为四十人，已有三十二人报名，还剩几个名额？', 'need': 0},
    {'text': '信封上写着收件人是周女士，收件人是谁？', 'need': 0},
    {'text': '阅读：雨停了，地面仍然潮湿。根据这句话，地面是什么状态？', 'need': 0},
    {'text': '订单包含一本笔记本和一把尺子，列出订单中的物品。', 'need': 0},
    # Non-retrieval requests
    {'text': '请写一句适合毕业典礼的祝福。', 'need': 0},
    {'text': '把数字 1、4、2 按从小到大的顺序排列。', 'need': 0},
    {'text': '暂停回答，不用查阅任何历史记录。', 'need': 0},
    {'text': '给虚构咖啡馆设计一个名字。', 'need': 0},
]

TEST = [
    # Prior personal experience
    {'text': '我过去采用过的哪种学习方法最有效？请据此调整复习方案。', 'need': 1},
    {'text': '我曾经说过，是什么促使我开始练习绘画的？', 'need': 1},
    {'text': '结合我之前讲述的家庭情况，分析这次分工可能遇到的阻碍。', 'need': 1},
    {'text': '把我先前提到的宠物就医过程按发生顺序整理出来。', 'need': 1},
    # General knowledge
    {'text': '月球自身会发出可见光吗？', 'need': 0},
    {'text': '通常用什么单位衡量电流？', 'need': 0},
    {'text': '为什么金属勺放进热汤里会逐渐变热？', 'need': 0},
    {'text': '分数二分之一和四分之一哪个更大？', 'need': 0},
    # Prompt-contained information
    {'text': '小船上午航行十二公里，下午航行八公里，这一天共航行多远？', 'need': 0},
    {'text': '通知写着展览因维修推迟，请根据通知说明推迟的原因。', 'need': 0},
    {'text': '包装标注净含量五百克，两个这样的包装共重多少克？', 'need': 0},
    {'text': '阿遥说：“如果下雨就带伞。”现在正在下雨，按她的计划应带什么？', 'need': 0},
    # Non-retrieval requests
    {'text': '请生成一段描述海边黄昏的文字。', 'need': 0},
    {'text': '我只是想说声谢谢。', 'need': 0},
    {'text': '将“风吹过树梢”改写成疑问句。', 'need': 0},
    {'text': '请忽略我的个人背景，给出一个通用的放松练习。', 'need': 0},
]
