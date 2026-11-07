"""CPU-only, streaming TRAIN corpus review; writes only P27-corpus-review.json.

Run with D:/SPORA/.venv/Scripts/python.exe scripts/audit_p27_corpus.py.
--collect saves an explicitly unfinished sample for annotation; default requires
the fixed, hash-bound annotations below. No held-out text or model is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/P27-corpus-review.json"
SEED = 20261108
PER_BUCKET = 4
FILES = ("pretrain-train.jsonl", "dialogue-train.jsonl")
# Labels refer to the whole record, except C (response prose) and E (speaker).
RUBRIC = {
    "C": ("complete_sentence_prose", "主要回答由语法可读、语义闭合的中文句子组成；标题和完整算式允许存在。代码/关键词/诗行主导或明显病句记否。不是逐句通过率，也不保证事实正确。"),
    "N": ("basic_narrative_content", "主体包含连贯、容易理解的日常描写或故事叙述；允许外围有生成/改写要求。区别于独立文章及步骤列表。"),
    "P": ("standalone_basic_prose", "整条可直接作为基础叙述文章使用，没有混入任务指令、问答串或明显内容缺陷。"),
    "B": ("basic_curriculum_usable", "适合基础句法或短任务课程：日常叙述、基础算术、简短澄清等，关系清楚且未发现实质缺陷；通常不超过600字。仍需正常的角色/格式处理，非事实核验认证。"),
    "T": ("template_or_padding", "存在可复用的空泛总括、广告修辞、冗余开场收尾、重复答案或重复清单骨架。仅有编号、正常礼貌或必要计算步骤不算。"),
    "X": ("visible_truncation", "原文中可见句子半截、未完成列表或未闭合输出；缺题干、短于字数要求、代码结尾和自然标题不单凭形式判截断。"),
    "E": ("unsupported_personal_experience", "回答者把没有上下文依据的生活事件或身体活动说成自身实际经历；不计用户陈述、明确故事/角色扮演或本轮确实完成的写作。"),
    "Q": ("substantive_quality_concern", "观察到逻辑/任务错误、病句、乱码、题干损坏、未经支持的具体断言，或需核实的事实疑点；不等于全部已证伪。单独的长篇、专业性、套话不自动记为缺陷。"),
    "R": ("unmarked_prompt_response_join", "预训练文本中的任务与回答/后续问题直接相接，缺少明确角色标记；带人物姓名或明确引号对话不算。"),
    "H": ("specialist_or_abstract", "主要要求专业代码、医学、历史专名、行业知识或抽象哲学；表示当前基础课程适配问题，不自动认定内容错误。"),
}

ANNOTATIONS = {
    "pretrain-train.jsonl": [
        (40654, "CQRH", "将管道称为点滴袋，医疗定义混淆；专业内容需核实。"),
        (43956, "CTR", "情感分类可理解，但对一个短句反复解释并添加总括；非叙述文章。"),
        (98627, "QRH", "Python代码被压平，for/if缩进与换行丢失；不是句尾截断。"),
        (144068, "QRH", "医学摘要与关键词标签相接，术语和引文数字密集；不构成完整句子教材。"),
        (245527, "CNBR", "女孩成长的简单概括，完整且与题干一致；仍含概括指令。"),
        (284579, "QRH", "政治解释有极左等未经支持归因，且规劝政府开支等搭配不通顺。"),
        (315857, "CQR", "瑜伽摘要中的流啊瑜伽等源文疑似错误，回答未纠正。"),
        (350761, "CBR", "五加三的语句与答案一致，适合基础算术短任务。"),
        (489664, "CQR", "故事虽明确为虚构，不算亲历；神经元在森林生活、与生态关系的设定混乱，角色串接。"),
        (524178, "CTQR", "多个无关问答拼接；将电视剧摩登家庭推荐为电影，且技术风险结论绝对化。"),
        (563516, "CQR", "含存疑字数/敏感词计数等处理残留，选项缺失；虽要求补充却又猜测公文惯例。"),
        (611833, "QR", "艾米滑雪题干语义破碎、夹西文，答案仍解释动态变量。"),
        (653553, "CQ", "引号分开的日常订餐对话，不属亲历；尚无实际地址即声称处理订单，流程信息不足。"),
        (656013, "CTQRH", "旅游宣传与抽取问答串接，把华尔街整体当建筑；含泛化夸赞。"),
        (670546, "CQRH", "公司总部的楼名、地址及现状有疑点，缺日期；不适合未经核实作基础事实。"),
        (689212, "CTQR", "反复使用提升形象/稳定性等骨架，前脸样式提升行驶稳定性缺乏依据。"),
        (754075, "CNR", "端午习俗介绍及改写提供基础描写；整条重复原文/改写，不是独立文章；甜粽描述不应泛化所有粽子。"),
        (759394, "CQR", "植物养护有一次两次无周期、浇水深度不超过根部两倍等不可靠具体指导。"),
        (819907, "TQRH", "人参先天地生长环境为病句；对心梗/高血压先推荐后慎用，功效断言与剂量需专业核实。"),
        (837401, "CTQRH", "傲慢与偏见分析出现逊意妙、弗吉尼亚州等明显情节疑点；主题反复铺陈。"),
        (866516, "C", "明确虚构且带姓名的医生患者对话，不是回答者实际经历；具体手术时长是情景设定，不用于医学事实教学。"),
        (873322, "CB", "人物标记清楚，机房设备更新的简短日常交流；未发现实质问题。"),
        (925089, "CBR", "最大最小值正确，比较过程可读；步骤重复是必要教学，不单记套话。"),
        (934261, "CTR", "七次类似未提供资料收尾，信息增量很低；拒绝补造事实的方向是合理的。"),
        (998191, "CTQR", "模板新闻扩写为据报道/专家研究等未经支持的现实断言，未显式标成虚构报道；不属第一人称经历。"),
        (1011686, "CNBR", "明确要求不存在的场景，外星生物描写连贯；幻想不作为事实或个人经历。"),
        (1017729, "CQRH", "把约翰哈佛捐赠者身份混作创校者，1865改名等说法需纠正；无外部逐项查证。"),
        (1052902, "CTRH", "五轮营销问答重复受众/互动/监测骨架，篇幅长；未标角色。"),
        (1056420, "CTQRH", "乳腺癌最新进展无日期且概念泛化，如较高的免疫学特征；不应作当前医疗事实。"),
        (1057146, "CTR", "化学品防护三轮多次重复保护自己/咨询专业人士；原文完整，没有可见截断。"),
        (1072614, "CTQR", "故事和技术说明拼接，一家三口却称孩子们；健康诊断能力先夸大后加限定。"),
        (1073903, "TQR", "英语歌词占显著篇幅，章节粘连；钢琴归为弦乐器的表述过简，后续文化清单反复。"),
        (1095176, "CTRH", "技术史文章、抽取、总结重复同一阶段框架；长而专业，不作逐项史实核验。"),
        (1099698, "CTQR", "气候文章及抽取复述；龙卷风等归因未经限定，交通建议句子搭配含混。"),
        (1113357, "CTR", "节庆修辞按青崖/白水意象反复铺展；形式完整但不适合基础具体叙述。"),
        (1138739, "CRH", "因式分解正确且算式闭合；公式结尾不算截断，但非基础日常叙述。"),
        (1195205, "CQR", "没有日期或工具依据却给出北京明天天气和温度。"),
        (1210130, "CQR", "童话开头、品牌、恐龙问答混杂；生成品牌名却列既有品牌，任务理解有疑点。"),
        (1212956, "CQR", "爱情小说推荐与作品类型/情节不够匹配，概括需复核。"),
        (1215051, "CTR", "名画介绍后每轮追加服务性问句与收尾；完整但多轮边界丢失。"),
    ],
    "dialogue-train.jsonl": [
        (11966, "CTH", "意识与现实等抽象概念反复铺陈；不含具体自身生活事件。"),
        (20019, "CTQH", "抽象自我议论加心灵鸡汤；海明威引语来源可疑，未经核实。"),
        (36703, "CTH", "记忆云的前两轮高度近义复述，仍是条件性讨论，不据此判事实错误。"),
        (48662, "CQ", "题干只给A到中间点距离，回答擅自设三个中间点到B均为5，结论无依据。"),
        (74442, "CTH", "技术伦理及宁静生活两套清单，话题转移来自用户；泛化收尾不算截断。"),
        (76552, "CTH", "两段自我边界说明及重复邀问；今天没有吃为含混否认，保守不算虚构生活事件。"),
        (113265, "CTH", "国际业务十项清单和总括高度模板化；未独立查证行业内容。"),
        (151313, "QH", "初版切片方向与说明相反，修正版空列表会除零；代码主导，未算完整中文句子教材。"),
        (170363, "CTQ", "题干夹homes/aturalsp及农药瓶太阳能板，回答顺着坏题干泛化说明。"),
        (183130, "CQ", "亲属图错误地把爷爷、外公外婆串成上下代，图文关系不清。"),
        (187515, "CTQH", "IATA英文是国际航空运输协会，却被写成两个动物运输组织；书名翻译也不对应。"),
        (217340, "CTH", "签证八项建议和泛化收尾；应区分签证有效期与获准停留期，此处不作法律核验。"),
        (269691, "CTQH", "把使用未定义变量笼统列为语法错误，有语言依赖性；类别和结尾泛化。"),
        (280878, "CT", "月亮文化六项清单加宏观总结；可作较宽阅读，非基础短叙述。"),
        (332662, "CT", "声学定义和音乐影响清单后再重复总结；主题可读但冗余，长于基础短回答。"),
        (333618, "CTQH", "照单认同三面转/绕横轴产生光晕的错误前提，并给出混淆横轴和左右倾斜的操作。"),
        (344622, "CTQH", "手机题干损坏，通用恢复清单未澄清具体设备与模式，工具适用性未经支持。"),
        (349274, "CTQ", "题干含混且后续步骤展开为通用茶文化清单；一次性茶包一律可回收等建议欠条件。"),
        (381761, "CTQH", "将各地反歧视保障混列，未区分具体法域和法律覆盖范围。"),
        (401397, "CT", "招聘广告是用户明确代写，不属个人经历；六大抽象品质和宣传结尾冗长。"),
        (458240, "CTQH", "对1127错误及软件自带修复功能作未核实断言，随后给出通用排查清单。"),
        (498781, "CTQ", "未给资产数据却称详细评估，并无解释地用更远路线节省油费。"),
        (500669, "CTQH", "缺失选项和比较领域定义，仍猜历史/文学为第一大类。"),
        (515296, "CQH", "设seed却未传给RAND且缺SQL分号；2021更新时间属于移植身份说法，不计生活亲历。"),
        (560698, "CTQH", "宣称正文1001汉字与实际篇幅不符，照片/软件作为医学影像等表达有误。"),
        (576846, "CTQH", "要求不少于1000字却只给短文；句子自然结束，属任务未完成而非观察到截断。"),
        (587367, "CB", "面包车5辆、轿车20辆、共25辆，依据及计算齐全。"),
        (587473, "CQ", "小棒计算成立，但原题图形字符和空位损坏，需清洗题干；公式完整。"),
        (617160, "CB", "正负数和数轴解释正确完整，结尾判断符号不视作截断。"),
        (621659, "CT", "研究摘要五项清单及结尾概括；比基础日常句法课程更偏学术写作。"),
        (625558, "TQ", "鸟类说明循环同义，数千翼展/飞莺等措辞含混并夹英文。"),
        (671805, "CTQ", "缺原句却重复输出同一模板，且含replacement占位词；并非截断。"),
        (689600, "TQ", "没有提供原文仍写诗并自夸；要求两句却给四行，诗句语义不通。"),
        (711479, "CB", "按物理速度矢量恒定理解，加速度为零正确；若指仅速率不变则需另作澄清。"),
        (715916, "Q", "没有英文原文却自造五句翻译，一是苹果等表达和译义有误。"),
        (759231, "CB", "识别缺少变量和需求，明确请求补充；合理澄清，不计缺题干错误回答。"),
        (766303, "CTQ", "购物付款问题答成预算建议，称买更贵替代品节省开支，且把携带500元当剩余500元。"),
        (874339, "CT", "承认缺实时信息并给查询渠道；较长且额外邀问，不声称已经查到数据。"),
        (890019, "CB", "简短说明无法取得实时天气新闻并提示查询，界限清楚。"),
        (890350, "CB", "简短说明缺实时温度/汇率；未提供摄氏度，不能直接算华氏度，无编造数值。"),
    ],
}

SAMPLE_FINGERPRINTS = {
    "pretrain-train.jsonl": "810609799d67500d5c56f0b7bc079a1751df5cd14fd305410e8874c62a8f1a84",
    "dialogue-train.jsonl": "410be57277a4a2a340e35244837679cfa2d0bb5c671d2891760ccb066803af17",
}


def small_json(relative):
    path = ROOT / relative
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def scan(name, manifest):
    path = ROOT / "runs/conversation-data-v5" / name
    before = path.stat()
    digest = hashlib.sha256()
    counts = Counter()
    lengths = Counter()
    turns = Counter()
    reservoirs = {b: [] for b in range(10)}
    rngs = {b: random.Random(f"{SEED}:{name}:{b}") for b in range(10)}
    total_chars = 0
    max_line_bytes = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            max_line_bytes = max(max_line_bytes, len(raw))
            row = json.loads(raw)
            bucket = row["source_byte_bucket"]
            if type(bucket) is not int or bucket not in reservoirs:
                raise ValueError(f"Unexpected bucket at {name}:{line_number}")
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string text at {name}:{line_number}")
            counts[bucket] += 1
            size = len(text)
            total_chars += size
            lengths[next((limit for limit in (128, 256, 512, 1024, 2048)
                          if size <= limit), "over2048")] += 1
            if name.startswith("dialogue"):
                turns[sum(m["role"] == "assistant" for m in row["messages"])] += 1
            n = counts[bucket]
            slot = n - 1 if n <= PER_BUCKET else rngs[bucket].randrange(n)
            if slot < PER_BUCKET:
                sample = {
                    "file_line_1based": line_number,
                    "source_line_as_stored": row["source_line"],
                    "bucket": bucket,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                }
                if "messages" in row:
                    sample["messages"] = row["messages"]
                if n <= PER_BUCKET:
                    reservoirs[bucket].append(sample)
                else:
                    reservoirs[bucket][slot] = sample
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert digest.hexdigest() == manifest["files"][name], f"Hash changed: {name}"
    kind = name.split("-")[0]
    assert sum(counts.values()) == manifest["stats"][f"{kind}_train_documents"]
    assert all(len(reservoirs[b]) == PER_BUCKET for b in range(10))
    samples = [s for b in range(10) for s in sorted(
        reservoirs[b], key=lambda x: x["file_line_1based"])]
    for i, sample in enumerate(samples):
        sample["review_index"] = i
    return {
        "path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size,
        "records": sum(counts.values()), "bucket_counts": dict(counts),
        "character_length_bins_inclusive_upper": dict(lengths),
        "character_count_including_role_markers": total_chars,
        "assistant_turn_histogram": dict(turns),
        "max_input_line_bytes": max_line_bytes, "samples": samples,
    }


def summarize(corpus):
    samples = corpus["samples"]
    metrics = sorted(samples[0]["review"]["labels"])
    result = {}
    for metric in metrics:
        positives = sum(s["review"]["labels"][metric] for s in samples)
        weighted = 0.0
        variance = 0.0
        per_bucket = {}
        for b in range(10):
            group = [s for s in samples if s["bucket"] == b]
            n = len(group)
            count = sum(s["review"]["labels"][metric] for s in group)
            p = count / n
            population = corpus["bucket_counts"].get(b, corpus["bucket_counts"].get(str(b)))
            weight = population / corpus["records"]
            weighted += weight * p
            variance += weight ** 2 * (1 - n / population) * p * (1 - p) / (n - 1)
            per_bucket[b] = count
        result[metric] = {
            "positive": positives, "sample_n": len(samples),
            "unweighted_percent": round(100 * positives / len(samples), 2),
            "document_weighted_percent": round(100 * weighted, 2),
            "design_standard_error_percentage_points": round(100 * math.sqrt(variance), 2),
            "positive_by_bucket": per_bucket,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--reuse-sample", action="store_true",
                        help="Reuse the saved sample after annotation; no corpus rescan")
    args = parser.parse_args()
    if args.reuse_sample:
        report = json.loads(OUTPUT.read_text(encoding="utf-8"))
        assert report["method"]["seed"] == SEED
    else:
        manifest, manifest_hash = small_json("runs/conversation-data-v5/manifest.json")
        lineage, lineage_hash = small_json("runs/conversation-data-v5/lineage-audit.json")
        provenance, provenance_hash = small_json("runs/chinese-prose-material/provenance.json")
        candidate_sample, candidate_hash = small_json("runs/chinese-prose-material/quality-sample.json")
        report = {
            "schema_version": 1, "status": "sampled_pending_manual_review",
            "workspace": str(ROOT),
            "scope": {"train_only": True, "heldout_text_read": False,
                      "gpu_used": False, "training_started": False,
                      "worker_threads": 1, "stdlib_only": True,
                      "leakage_audit_repeated": False},
            "method": {"seed": SEED, "per_source_byte_bucket": PER_BUCKET,
                       "sample_n_per_partition": 40, "total_train_sample_n": 80,
                       "selection": "Independent Algorithm-R reservoirs in each of ten stored source_byte_bucket strata; RNG seed string = seed:filename:bucket; physical lines streamed once.",
                       "estimand": "Document prevalence within each raw TRAIN JSONL, weighted by full-scan stratum counts. Not token exposure or prevalence in the encoded subset.",
                       "uncertainty": "One reviewer; n=4 per stratum. Standard errors are design plug-in estimates and understate uncertainty for all-zero/all-one strata; no confidence interval or zero-risk claim. Labels overlap; no combined partition rate without training exposure weights."},
            "existing_evidence": {
                "manifest_sha256": manifest_hash, "lineage_sha256": lineage_hash,
                "lineage_passed_as_reported": lineage["passed"],
                "lineage_scope_as_reported": lineage["scope"],
                "prior_reviews": ["reports/P23-pretrain-quality-audit.json",
                                  "reports/P23-multiturn-quality-review.json",
                                  "reports/P23-prose-source-review.json"],
                "training_history_summaries": ["reports/P23c-progress.md", "reports/P23d-progress.md", "reports/P23e-progress.md"],
                "train_manifest_stats": {k: v for k, v in manifest["stats"].items()
                                         if "train" in k and "test" not in k},
                "preparation_stats": manifest["preparation_stats"],
            },
            "corpora": {},
            "prose_candidate": {"provenance": provenance,
                                "provenance_sha256": provenance_hash,
                                "existing_sample_sha256": candidate_hash,
                                "existing_sample": candidate_sample,
                                "new_source_scan": False,
                                "scope": "Existing documents and previously saved samples only; no new representative estimate for the candidate."},
        }
        for name in FILES:
            report["corpora"][name] = scan(name, manifest)
            print(f"Scanned {name}: {report['corpora'][name]['records']} records", flush=True)
    if not args.collect:
        apply_review(report)
        report["status"] = "review_complete_no_training"
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(OUTPUT), flush=True)


def apply_review(report):
    report["review_rubric"] = {name: definition for name, definition in RUBRIC.values()}
    report["reviewer"] = "Single development-agent qualitative review of full sampled records; not independent human blind annotation."
    for name, corpus in report["corpora"].items():
        samples = corpus["samples"]
        fingerprint = hashlib.sha256("\n".join(
            s["text_sha256"] for s in samples).encode()).hexdigest()
        assert fingerprint == SAMPLE_FINGERPRINTS[name], "Sample changed; re-review required"
        assert len(samples) == len(ANNOTATIONS[name]) == 40
        for sample, (source_line, codes, note) in zip(samples, ANNOTATIONS[name]):
            assert sample["source_line_as_stored"] == source_line
            assert hashlib.sha256(sample["text"].encode("utf-8")).hexdigest() == sample["text_sha256"]
            assert set(codes) <= set(RUBRIC)
            sample["review"] = {
                "labels": {metric: code in codes for code, (metric, _) in RUBRIC.items()},
                "note": note,
            }
        corpus["sample_fingerprint"] = fingerprint
        corpus["manual_statistics"] = summarize(corpus)
    # These two rows were observed during schema inspection, not reservoir draws.
    # Read at most two records; never count them toward prevalence estimates.
    targeted = []
    with (ROOT / "runs/conversation-data-v5/dialogue-train.jsonl").open(
            encoding="utf-8") as stream:
        for expected in (465, 528):
            row = json.loads(next(stream))
            assert row["source_line"] == expected
            targeted.append({
                "source_line_as_stored": expected,
                "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "first_assistant_response": next(m["content"] for m in row["messages"]
                                                 if m["role"] == "assistant"),
                "finding": "回答者无依据认领工作学习/喝咖啡/与朋友聊天等实际生活经历；非显式角色扮演。",
            })
    report["targeted_findings_not_in_probability_sample"] = {
        "selection": "First two dialogue TRAIN rows, read originally to inspect schema; purposive, not random.",
        "unsupported_personal_experience": targeted,
        "inference": "确证当前原始训练文件仍有此类内容；不能据这两条估计比例，也未判断其是否进入长度过滤后的编码子集。",
    }
    stats = report["existing_evidence"]["train_manifest_stats"]
    pretrain = report["corpora"]["pretrain-train.jsonl"]
    dialogue = report["corpora"]["dialogue-train.jsonl"]
    old_bucket_sum = sum(stats[f"pretrain_train_bucket{b}"] for b in range(10))
    report["manifest_interpretation"] = {
        "observed_pretrain_bucket_counts_match_preparation_stats": all(
            pretrain["bucket_counts"].get(b, pretrain["bucket_counts"].get(str(b))) ==
            report["existing_evidence"]["preparation_stats"][f"train_bucket{b}"]
            for b in range(10)),
        "inherited_stats_pretrain_bucket_sum": old_bucket_sum,
        "current_pretrain_count": pretrain["records"],
        "finding": "manifest.stats中的旧预训练区段计数合计135230，而现文件1085328条；preparation_stats与本轮实扫吻合。加权采用本轮计数，不把旧区段统计混入v5分母。",
        "dialogue_encoded": stats["dialogue_train_encoded"],
        "dialogue_overlength_excluded": stats["dialogue_train_overlength"],
        "overlength_percent_of_raw": round(100 * stats["dialogue_train_overlength"] / dialogue["records"], 2),
        "overlength_interpretation": "这是清单声明的长度筛除，不是原文截断率；本轮未读取编码文件，也未推断训练器窗口截取行为。",
        "raw_dialogue_single_turn_count": dialogue["assistant_turn_histogram"].get(
            1, dialogue["assistant_turn_histogram"].get("1")),
        "raw_dialogue_single_turn_percent": round(100 * dialogue["assistant_turn_histogram"].get(
            1, dialogue["assistant_turn_histogram"].get("1")) / dialogue["records"], 2),
        "unconsumed_v5_claim": "用户说明246M词元尚未使用；本审计不检查训练谱系或曝光记录，v5文本与早先子集有继承关系，不能把全部246M视作独立新增词元。",
    }
    candidate = report["prose_candidate"]
    candidate["document_review"] = {
        "existing_saved_samples_reviewed": 12,
        "estimated_population_quality_rate": None,
        "observations": [
            {"source_row": 141852, "issue": "公式及变量丢失，正文关系不完整，直接排除。"},
            {"source_row": 210408, "issue": "末段出现先知预言和的等病句，需修订或排除。"},
            {"source_row": 249399, "issue": "传记含院士。，也等标点损坏，专名与日期密集。"},
            {"source_row": 26271, "issue": "标题日期只剩16日，原始信息缺失。"},
            {"source_row": 180115, "issue": "引文编号73粘连到正文，不宜直接学习。"},
            {"source_row": 197521, "issue": "thumb残留；移除后地理/地质段落可作为候选，仍须核实与切分。"},
        ],
        "visible_cleanup_or_corruption_sample_count": 6,
        "other_sample_characteristics": "其余条目也主要是人物、历史、体育、宗教、时区、商业模式或科学概念；文章体裁不等于适合8.6M的低难度叙述。",
        "decision": "可作为有出处的普通文章来源候选；暂不全量投喂，也未估计可回收产量。不能从现存12条外推全库合格率。",
        "preparation_proposal": [
            "保留集合出处、固定修订、来源字段及提供方CC-BY-SA-3.0声明；缺逐篇URL/标题字段的限制写入新清单。",
            "优先筛选日常物体、自然现象、生活过程和可理解故事；历史人名/日期目录、公式缺项、残留标记、外语密集条目降权或排除。",
            "按完整句/段切分，建议80至300个中文字符为主体、最长约500字符；再用现有8192词表核对长度，字符数不能代替词元数。",
            "保留标题和必要前文；不可把中间残段当自足文章，不将第三人称传记转换为模型自身经历。",
            "若以后确实准备新语料，由负责隔离的流程执行既定重叠检查；本次没有打开任何保留分区，也没有执行新泄漏审计。",
        ],
    }
    report["recommendations"] = {
        "decision": "v5可用于受控广覆盖试段；不支持把全量v5等同优质普通叙述后直接提高到70%。维基来源尚需清理。",
        "status": "Hypotheses and operational proposals, not an experimentally established optimum; no training was run.",
        "requested_start": "用户计划从P23f-small2000继续，保持自研约8.6M、现有词表及原个体；本轮只提供数据建议。",
        "immediate_next_segment": {
            "unit": "有效损失目标词元的预算份额，不是文件条数或批次数；重点回放计入所属份额，不重复加算。",
            "v5_pretrain_percent": 40,
            "existing_broad_instruction_percent": 55,
            "reviewed_dialogue_percent": 4,
            "small_curriculum_percent": 1,
            "availability": "v5与既有广覆盖教学可用。4%对话槽须来自已筛选来源，不直接将当前原始对话全量视为合格；若没有合格子集，将这4%暂并入广覆盖教学，即40/59/0/1。",
            "scope": "这是控制新分布曝光的短试段，不是正式质量认证。40%并未由样本统计求得最优值，也不证明优于20%。",
            "checkpoint_budget_proposal": "首个检查点以新增约100万有效目标词元为上限（约400k/550k/40k/10k）；只作实验规模建议，主流程按已批准验证方案决定是否延长。",
        },
        "conditional_70_percent_text_recipe": {
            "unit": "effective_loss_target_tokens",
            "clean_ordinary_prose_percent": 50,
            "screened_v5_percent": 20,
            "broad_instruction_percent": 25,
            "grounded_dialogue_percent": 4,
            "small_curriculum_percent": 1,
            "not_ready": "普通文章和有依据对话必须先满足质量门槛；当前没有新建这些数据，本建议不自动授权本脚本扩大范围。",
        },
        "batch_vs_target_warning": {
            "source": "reports/P23d-progress.md",
            "observed_plain_batch_percent": 20,
            "observed_plain_targets": 819200,
            "observed_total_targets": 1456296,
            "observed_plain_target_percent": round(100 * 819200 / 1456296, 2),
            "interpretation": "D的20%文章批次实际贡献约56%目标。新训练器批长、掩码、打包不同，不能沿用此比例换算；应记录实际各来源目标数并校准批次安排。",
        },
        "filtering_priorities": [
            "首先去除坏题干、明显任务/计算错误、空输入却猜测、角色身份和无依据生活亲历；不只做关键词黑名单。",
            "将明确的任务/回答边界恢复到指令格式；无法可靠恢复的串接记录，不当独立叙述供给。",
            "降低重复开场结尾、长清单、近义多轮和抽象自我议论的曝光；不过度惩罚正常礼貌或必要计算步骤。",
            "按来源和长度记录采样份额；同文档/近义模板限制重复回放，小教材保持1%有效目标上限。",
            "可见截断为零不证明打包后句子完整；新训练器应由主流程检查文档边界及末句/EOS保留，本任务不检查或修改训练器。",
        ],
        "validation_proposal": "由主流程使用既定公开开发协议、相同解码与目标预算，核对基础句子、上下文依据、循环及原个体保持；同时观察旧开发损失。不要仅凭训练损失下降或结束率改善宣称能力提升，不打开未公开验收题。",
    }
    report["observation_vs_inference"] = {
        "observed": [
            "两份原始训练文件的SHA256与清单一致；条数和区段权重来自单次流式扫描。",
            "每分区40条、十区段各4条，共80条固定种子样本全部人工式语义评阅；分类和逐条理由可追溯。",
            "完整句子与适用基础教材不是同一指标；独立叙述、模板、语义缺陷分别统计。",
            "随机样本没有明确句中截断或未支持生活亲历；另两条定向记录确认后者仍然存在。",
            "现有普通文章说明及12条旧样本提示至少六条有清理/损坏问题；未新增文章抽样。",
        ],
        "hypotheses": [
            "更多短而自足的普通文章可能比扩大拼接问答更有利于8.6M的基础语言学习，但本轮没有模型实验。",
            "模板和低质事实可能挤占小模型容量；无法仅从语料审计判定它们是既往训练失败的主要原因。",
            "40/55/4/1与条件性50/20/25/4/1是可验证起点，不是统计估计或最优性结论。",
        ],
        "limitations": [
            "每层仅4条，单评阅者标签主观，稀有问题很可能漏检；零观察不等于零发生率。",
            "按文档加权，不按词元或真实训练曝光加权；原始对话样本包含可能被长度筛除的记录。",
            "事实疑点未逐项外部核验，质量问题比例混合可见错误与需核查断言。",
            "没有量化全库近义重复、可回收普通文章规模或对话编码子集的实际问题率。",
            "继承已有泄漏初审结论及其有限范围，本次不提供新的隔离证明。",
        ],
    }


if __name__ == "__main__":
    main()
