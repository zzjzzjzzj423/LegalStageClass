"""第二阶段：把文件名称组进一步归入 8 大类、22 个内容领域。

复用 prepare_stard.py 的第一阶段分组函数，不重复实现名称前缀解析。
当前输入为 1,258 个文件组，允许一个文件属于多个领域；生成分类结果及验收报告。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from data.processed.prepare_stard import load_stard_corpus, group_articles_by_document
from legal_stage_class.util import write_jsonl

OUT = ROOT / "data/processed"
RAW = ROOT / "data/raw/STARD"
SEED = 20261108
PARENTS = {
    "G01": "民事生活与商事活动", "G02": "劳动与社会保障",
    "G03": "刑事犯罪与刑罚", "G04": "诉讼与纠纷处理",
    "G05": "行政、国家事务与公共安全", "G06": "金融与财政税务",
    "G07": "土地、环境与基础设施", "G08": "社会公共服务",
}
# 第二阶段按文件名称中的内容关键词分类；标准答案不参与制定标签。
DEFINITIONS = [
    ("D01", "G01", "民事综合、合同与财产权利", r"民法典|民法通则|民法总则|合同|担保|债权|债务|物权|租赁|保管|定作|承揽|保证|抵押|质押|侵权|人身损害|精神损害|拍卖|不动产|动产|民事责任|诉讼时效"),
    ("D02", "G01", "婚姻家庭与继承", r"婚姻|家庭|离婚|夫妻|继承|收养|抚养|扶养|赡养|遗产"),
    ("D03", "G01", "公司企业与破产", r"公司|企业|合伙|破产|个体工商|市场主体|合作社|外商投资|外资|投资者|投资保护|鼓励投资|经营者集中"),
    ("D04", "G01", "知识产权", r"知识产权|著作权|商标|专利|集成电路|植物新品种|计算机软件|技术合同|信息网络传播权|特殊标志|世界博览会标志|奥林匹克标志"),
    ("D05", "G01", "市场秩序与消费者保护", r"消费者|消费品|产品质量|反垄断|垄断|反不正当|竞争|价格|计量|标准化|认证|直销|传销|广告|商业特许|电子商务|烟草|经营者集中|暴利|地区封锁|无证无照|食盐专营|产品召回|修理更换退货"),
    ("D06", "G02", "劳动就业与社会保障", r"劳动|劳务|工资|工伤|职工|就业|社会保险|社会保障|失业|养老|工会|人事争议|人力资源|职业病|童工|工人|技师|退休|放假"),
    ("D07", "G03", "刑事犯罪与刑罚", r"刑法|刑事|刑罚|犯罪|罪|减刑|假释|监狱|社区矫正|看守所|拘留所|自首|立功"),
    ("D08", "G04", "诉讼仲裁执行与司法组织", r"诉讼|仲裁|执行异议|执行程序|强制执行|人民法院执行|法院办理执行|民事执行|执行工作|审判|庭审|合议庭|法官|检察官|人民陪审|律师|法律援助|公证|司法鉴定|人民调解|司法确认|再审|立案|裁判|管辖|送达|证据|审限|开庭|督促程序|法院组织|检察院组织|司法强制"),
    ("D09", "G05", "行政程序与政府管理", r"行政许可|行政处罚|行政复议|行政执法|行政强制|行政机关|政府信息公开|公务员|事业单位|机关事务|信访|政府采购|监察|统计|普查|政务|公文|机构编制|法规规章|规章制定|法规制定|罚款|集会|游行|示威|行政区域|行政区划"),
    ("D10", "G05", "公共安全与应急管理", r"安全|消防|应急|危险|爆炸|爆破|枪支|弹药|烟花|特种设备|防震|地震|气象|救灾|自然灾害|突发事件|禁毒|戒毒|治安|警察|公安|恐怖|间谍|情报|国家秘密|海警|刀具|保安|核材料|监控化学品|天气|易制毒"),
    ("D11", "G05", "国防军事与国家制度", r"国防|军事|军人|军官|军队|兵役|预备役|军衔|军委|解放军|驻军|宪法|人民代表大会|立法|国旗|国徽|国歌|勋章|国家荣誉|国家赔偿|民族|特别行政区|香港|澳门|国家机构|戒严|反分裂|人民防空|民兵|征兵|军服|军品|军用|军工|国务院组织"),
    ("D12", "G05", "对外事务贸易与出入境", r"外交|领事|外国|国籍|护照|出境|入境|出入境|对外|进出口|海关|贸易|关税|反倾销|反补贴|保障措施|国际条约|国际刑事|国际司法|国际货物|国际贸易|引渡|使馆|驻外|条约|领海|国界|大陆架|出口|往来台湾|涉外民事"),
    ("D13", "G06", "金融证券与保险", r"银行|金融|证券|期货|保险|外汇|信托|票据|支付|融资|存款|借贷|借款|贷款|征信|保理|洗钱|人民币|金银|集资|股票|现金|储蓄"),
    ("D14", "G06", "财税会计与审计", r"税|财政|会计|审计|预算|发票|国库|彩票|国债|资产评估|财务|金库"),
    ("D15", "G07", "土地农业与农村", r"土地|农业|农村|农民|农田|种子|农药|肥料|畜|兽|饲料|渔业|水产|草原|林业|森林|农机|农用|乡村|乡镇|耕|粮食|储备粮|病虫害|屠宰"),
    ("D16", "G07", "环境资源与能源", r"环境|污染|生态|资源|矿|能源|电力|电网|煤|石油|天然气|节能|节约能源|水土|水法|水文|取水|河道|河流|水库|水利|海域|海洋|海岛|防洪|防沙|荒漠|野生动物|野生植物|自然保护|排水|污水|湿地|长江|黄河|测绘|测量标志|地质|循环经济|清洁生产|电缆|管道|防汛|供用水|地下水|太湖|抗旱|臭氧|地图|废弃电器|蓄滞洪|排污"),
    ("D17", "G07", "住房物业与工程建设", r"房|物业|建筑|建设工程|工程建设|拆迁|土地征收|征收补偿|住房|城乡规划|工程质量|招标|投标|建设项目|村庄和集镇规划"),
    ("D18", "G07", "交通运输与出行", r"交通|运输|道路|公路|铁路|航空|航道|航运|船|港口|海商|海事|车辆|机动车|驾驶|出租汽车|快递|邮政|机场|汽车|航行|飞行|航标"),
    ("D19", "G08", "医疗卫生与食品药品", r"医疗|医师|护士|卫生|健康|疾病|传染病|防疫|检疫|药|食品|疫苗|血|医院|诊疗|医患|计划生育|人口|母婴|化妆品|器官|艾滋病|尘肺病|食盐加碘"),
    ("D20", "G08", "教育科技文化与体育", r"教育|学校|教师|学位|学生|科研|科学|科技|科学技术|技术合同|农业技术|技术进步|文化|文物|博物馆|出版|图书|电影|广播|电视|音像|艺术|演出|体育|健身|奥林匹克|奥运|非物质|档案|考古|兴奋剂|影片|实验动物|印刷|工艺美术|教学|娱乐|地方志|办学|文盲|长城|古生物|幼儿园|语言文字"),
    ("D21", "G08", "网络数据与通信", r"网络|互联网|数据|电信|通信|计算机|密码|电子签名|无线电|信息安全|个人信息|信息服务|信息系统|信息公开"),
    ("D22", "G08", "民政社会事务与特殊群体", r"(?<!人)民政|社会团体|社会组织|基金会|慈善|捐赠|公益|救助|救济|抚恤|残疾|老年|未成年人|妇女|儿童|退役|军属|烈士|殡葬|地名|边界|居民|村民|户口|户籍|旅游|旅馆|殡|志愿|红十字|归侨|侨眷|收容|流浪|安置|烈属|优抚|宗教|居住证|风景名胜|导游|旅行社"),
]
# 名称有歧义时明确指定领域，例如文化遗产不属于婚姻继承；不依据标准答案修正。
OVERRIDES = {
    "中华人民共和国民法典": ["D01", "D02", "D17"],
    "中华人民共和国刑法": ["D07"],
    "宪法": ["D11"],
    "国家通用语言文字法": ["D20"],
    "全国人民代表大会组织法": ["D11"],
    "地方各级人民代表大会和地方各级人民政府组织法": ["D09", "D11"],
    "国务院组织法": ["D09", "D11"],
    "城市居民委员会组织法": ["D22"],
    "村民委员会组织法": ["D15", "D22"],
    "国务院关于鼓励台湾同胞投资的规定": ["D03", "D12"],
    "全国年节及纪念日放假办法": ["D06"],
    "国家赔偿法": ["D09", "D11"],
    "国务院行政机构设置和编制管理条例": ["D09"],
    "境外非政府组织境内活动管理法": ["D12", "D22"],
    "工业产品生产许可证管理条例": ["D05", "D09"],
    "工商行政管理所条例": ["D05", "D09"],
    "市场监督管理投诉举报处理暂行办法": ["D05", "D09"],
    "政府参事工作条例": ["D09"],
    "政府投资条例": ["D09", "D14"],
    "政府督查工作条例": ["D09"],
    "最高人民法院于适用财产刑若干问题的规定": ["D07", "D08"],
    "最高人民法院关于严格执行案件审理期限制度的若干规定": ["D08"],
    "最高人民法院关于公布失信被执行人名单信息的若干规定": ["D08"],
    "最高人民法院关于委托执行若干问题的规定": ["D08"],
    "最高人民法院关于审理伪造货币等案件具体应用法律若干问题的解释": ["D07", "D13"],
    "最高人民法院关于审理伪造货币等案件具体应用法律若干问题的解释（二）": ["D07", "D13"],
    "最高人民法院关于审理侵犯商业秘密民事案件适用法律若干问题的规定": ["D04"],
    "最高人民法院关于审理信用证纠纷案件若干问题的规定": ["D13"],
    "最高人民法院关于审理因垄断行为引发的民事纠纷案件应用法律若干问题的规定": ["D05"],
    "最高人民法院关于审理存单纠纷案件的若干规定": ["D13"],
    "最高人民法院关于审理抢劫案件具体应用法律若干问题的解释": ["D07"],
    "最高人民法院关于审理挪用公款案件具体应用法律若干问题的解释": ["D07"],
    "最高人民法院关于审理无正本提单交付货物案件适用法律若干问题的规定": ["D18"],
    "最高人民法院关于审理海上货运代理纠纷案件若干问题的规定": ["D18"],
    "最高人民法院关于审理涉台民商事案件法律适用问题的规定": ["D08", "D12"],
    "最高人民法院关于审理独立保函纠纷案件若干问题的规定": ["D01", "D13"],
    "最高人民法院关于审理行政协议案件若干问题的规定": ["D01", "D09"],
    "最高人民法院关于执行和解若干问题的规定": ["D08"],
    "最高人民法院关于认可和执行台湾地区法院民事判决的规定": ["D08", "D12"],
    "最高人民法院关于适用简易程序审理民事案件的若干规定": ["D08"],
    "最高人民法院关于限制被执行人高消费的若干规定": ["D08"],
    "机关团体建设楼堂馆所管理条例": ["D09", "D17"],
    "武器装备研制设计师系统和行政指挥系统工作条例": ["D11"],
    "行政事业性国有资产管理条例": ["D09", "D14"],
    "行政学院工作条例": ["D09", "D20"],
    "违反行政事业性收费和罚没收入收支两条线管理规定行政处分暂行规定": ["D09", "D14"],
    "重大行政决策程序暂行条例": ["D09"],
    "城市供水条例": ["D16", "D17"],
    "城市绿化条例": ["D16", "D17"],
    "城市节约用水管理规定": ["D16", "D17"],
    "城镇燃气管理条例": ["D10", "D16", "D17"],
    "人民法院在线调解规则": ["D08", "D21"],
    "北京市新技术产业开发试验区暂行条例": ["D03", "D20"],
    "合理化建议和技术改进奖励条例": ["D20"],
    "国际海运条例": ["D12", "D18"],
    "最高人民法院关于人民法院办理自赔案件程序的规定": ["D08", "D09"],
    "最高人民法院关于人民法院办理财产保全案件若干问题的规定": ["D08"],
    "最高人民法院关于人民法院民事调解工作若干问题的规定": ["D08"],
    "最高人民法院关于人民法院特邀调解的规定": ["D08"],
    "最高人民法院关于人民法院确定财产处置参考价若干问题的规定": ["D08"],
    "最高人民法院关于设立国际商事法庭若干问题的规定": ["D08", "D12"],
    "棉花质量监督管理条例": ["D05", "D15"],
    "武器装备质量管理条例": ["D11"],
    "非物质文化遗产法": ["D20"],
    "社会抚养费征收管理办法": ["D09", "D14", "D19"],
    "优化营商环境条例": ["D03", "D09"],
    "人类遗传资源管理条例": ["D19", "D20"],
    "卖淫嫖娼人员收容教育办法": ["D10", "D22"],
    "保障农民工工资支付条例": ["D06"],
    "最高人民法院关于适用《民法典》婚姻家庭编的解释（一）": ["D02"],
    "最高人民法院关于适用《民法典》继承编的解释（一）": ["D02"],
    "地方各级人民政府机构设置和编制管理条例": ["D09"],
}


def main():
    """先取得第一阶段文件组，再生成领域分类与离线覆盖报告。"""
    # 第一步：复用文件名称分组，当前语料得到 1,258 个文件组。
    corpus = load_stard_corpus(RAW / "corpus.jsonl")
    catalog, article_laws = group_articles_by_document(corpus)
    # 第二步：为文件组匹配一个或多个内容领域。
    domains = []
    domain_articles = {}
    domain_files = {}
    for domain_id, parent_id, name, pattern in DEFINITIONS:
        domains.append({"domain_id": domain_id, "parent_id": parent_id,
                        "parent_name": PARENTS[parent_id], "name": name, "title_pattern": pattern})
        domain_articles[domain_id] = set()
        domain_files[domain_id] = []
    files = []
    article_domains = {}
    unmatched = []
    for law in catalog:
        name = law["canonical_law_name"]
        title = name.replace("最高人民法院", "").replace("最高人民检察院", "")
        selected = []
        evidence = []
        if name in OVERRIDES:
            selected = OVERRIDES[name]
            evidence.append("exact_title_review")
        else:
            for domain_id, _, _, pattern in DEFINITIONS:
                match_title = title
                if domain_id == "D16":
                    match_title = title.replace("人力资源", "")
                if domain_id == "D15":
                    match_title = title.replace("农民工", "")
                if domain_id == "D13":
                    match_title = re.sub(r"社会保险|工伤保险|养老保险|失业保险|生育保险|医疗保险", "", title)
                match = re.search(pattern, match_title)
                if match:
                    selected.append(domain_id)
                    evidence.append(domain_id + ":" + match.group())
        if not selected:
            unmatched.append(name)
        entry = dict(law)
        entry["domain_ids"] = selected
        entry["evidence"] = evidence
        files.append(entry)
        for article_id in law["article_ids"]:
            article_domains[article_id] = selected
        for domain_id in selected:
            domain_articles[domain_id].update(law["article_ids"])
            domain_files[domain_id].append(law["law_id"])
    if unmatched:
        print(json.dumps({"unmatched": unmatched}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    write_jsonl(OUT / "file_domains.jsonl", files)
    by_id = {}
    for article in corpus:
        by_id[article["article_id"]] = article
    rng = random.Random(SEED)
    samples = []
    summary = []
    for domain in domains:
        domain_id = domain["domain_id"]
        candidates = []
        for law in files:
            if domain_id in law["domain_ids"]:
                candidates.append(law)
        for law in rng.sample(candidates, min(5, len(candidates))):
            article_id = rng.choice(law["article_ids"])
            article = by_id[article_id]
            samples.append({"domain_id": domain_id, "article_id": article_id,
                            "name": article["name"], "content": article["content"]})
        summary.append({"parent_id": domain["parent_id"], "parent_name": domain["parent_name"],
                        "domain_id": domain_id, "domain_name": domain["name"],
                        "file_count": len(domain_files[domain_id]),
                        "article_count": len(domain_articles[domain_id])})
    queries = json.loads((RAW / "queries.json").read_text(encoding="utf-8"))
    query_results = []
    union_cache = {}

    def candidates_for(domain_ids):
        key = tuple(domain_ids)
        if key not in union_cache:
            allowed = set()
            for domain_id in domain_ids:
                allowed.update(domain_articles[domain_id])
            union_cache[key] = allowed
        return union_cache[key]

    for query in queries:
        gold = set()
        for article_id in query["match_id"]:
            gold.add(str(article_id))
        missing = gold.difference(article_domains)
        if missing:
            raise ValueError(f"unknown gold IDs: {missing}")
        eligible = []
        for domain_id in domain_articles:
            if gold.intersection(domain_articles[domain_id]):
                eligible.append(domain_id)
        best_single = min(eligible, key=lambda d: (-len(gold.intersection(domain_articles[d])), len(domain_articles[d]), d))
        selected = tuple(eligible)
        allowed = candidates_for(selected)
        single_allowed = domain_articles[best_single]
        query_results.append({"query_id": str(query["query_id"]), "query": query["问题"],
                              "gold_article_ids": sorted(gold), "gold_informed_domain_ids": list(selected),
                              "gold_informed_candidate_articles": len(allowed),
                              "gold_informed_removed_articles": len(corpus) - len(allowed),
                              "gold_informed_missing_gold": sorted(gold.difference(allowed)),
                              "best_single_domain": best_single,
                              "single_candidate_articles": len(single_allowed),
                              "single_missing_gold": sorted(gold.difference(single_allowed))})
    single_missed = 0
    single_failed_queries = 0
    coverage_counts = Counter()
    for row in query_results:
        coverage_counts[len(row["gold_informed_domain_ids"]) ] += 1
        single_missed += len(row["single_missing_gold"])
        single_failed_queries += bool(row["single_missing_gold"])
    stats = {"corpus_articles": len(corpus), "document_groups": len(catalog),
             "parents": len(PARENTS), "domains": len(domains),
             "unclassified_articles": 0, "unclassified_files": 0,
             "multi_domain_files": sum(len(row["domain_ids"]) > 1 for row in files),
             "queries": len(queries), "gold_pairs": sum(len(row["gold_article_ids"]) for row in query_results),
             "gold_informed_missing_gold_pairs": sum(len(row["gold_informed_missing_gold"]) for row in query_results),
             "gold_informed_domain_count_distribution": dict(sorted(coverage_counts.items())),
             "gold_informed_mean_candidate_articles": statistics.mean(row["gold_informed_candidate_articles"] for row in query_results),
             "gold_informed_median_candidate_articles": statistics.median(row["gold_informed_candidate_articles"] for row in query_results),
             "gold_informed_mean_removed_articles": statistics.mean(row["gold_informed_removed_articles"] for row in query_results),
             "best_single_missing_gold_pairs": single_missed,
             "best_single_queries_with_missing_gold": single_failed_queries,
             "best_single_mean_candidate_articles": statistics.mean(row["single_candidate_articles"] for row in query_results),
             "seed": SEED, "live_jev_evaluated": False,
             "classification_basis": "corpus titles and explicit title overrides; no gold labels",
             "evaluation_basis": "union of all domain labels of the gold articles; not predicted routing"}
    stats["gold_informed_mean_reduction_percent"] = 100 * stats["gold_informed_mean_removed_articles"] / len(corpus)
    for filename in ("corpus.jsonl", "queries.json"):
        stats[filename + "_sha256"] = hashlib.sha256((RAW / filename).read_bytes()).hexdigest()
    report = ["# 数据集分类规则", "", "## 分类规则", "",
              "对象是 STARD 的 55,348 条候选条文，按现有名称前缀规则得到 1,258 个文件组，再归入 8 个大类、22 个具体领域。",
              "1. 复用 data/processed/prepare_stard.py 的 group_articles_by_document：去掉条文名称末尾的‘第几条／第几条之几’，剩余名称相同即为一个文件组。",
              "2. 判断领域时去掉最高人民法院、最高人民检察院这两个发布机关名称，避免把发布机关误当作内容领域；具体匹配词见data/processed/classify_domains.py。",
              "3. 按文件名称中的内容关键词匹配领域，一个文件可匹配多个领域，例如劳动争议司法解释归入劳动领域，涉及诉讼程序时也可归入纠纷处理。",
              "4. 对名称有歧义的文件明确指定归属，例如‘非物质文化遗产法’属于文化，而非婚姻继承；‘人民法院’中的连续字‘民法’不作为民事匹配依据。",
              "5. 以文件为单位继承领域，不拆分综合文件内部章节。《民法典》整份归入民事、婚姻、住房三个领域。",
              "6. 分类规则只使用候选文件名称，不使用问题标准答案。标准答案仅用于后面的覆盖核对和候选范围测算。", "",
              "## 验收口径", "",
              "保留原文件分组，按内容领域多标签归类；文件中的全部条文继承文件标签。没有拆分《民法典》等综合文件。分类使用名称规则及逐项修正，未用标准答案生成标签。",
              "所有标准答案仅用于下文的覆盖验收。领域归属未经法律专家逐条复核，结构性无遗漏不代表语义分类完全准确。",
              "选择领域时已知标准答案：合并所有标准条文的领域标签，再取这些领域下全部条文的并集。没有为了缩小范围而挑选更小的标签组合。这是利用标准答案做的离线范围测算，不是真实路由、检索 Recall 或 Jev 成绩。", "",
              "## 覆盖与范围测算", "",
              "| 检查项 | 结果 |", "| --- | ---: |",
              f"| 文件名称组 | {len(catalog):,} |",
              f"| 原始候选条文 | {len(corpus):,} |",
              "| 未分类文件 / 条文 | 0 / 0 |",
              f"| 同时归属多个领域的文件 | {stats['multi_domain_files']} |",
              f"| 核对的问题 / 标准条目配对 | {len(queries):,} / {stats['gold_pairs']:,} |",
              "| 标准答案领域并集遗漏标准条目 | 0 |",
              f"| 标准答案领域并集平均剩余候选条文 | {stats['gold_informed_mean_candidate_articles']:,.2f} |",
              f"| 标准答案领域并集平均少搜条文 | {stats['gold_informed_mean_removed_articles']:,.2f} |",
              f"| 标准答案领域并集平均范围缩减 | {stats['gold_informed_mean_reduction_percent']:.2f}% |",
              f"| 即使选择最佳单领域仍会漏答案的问题 | {single_failed_queries} |",
              f"| 最佳单领域遗漏的标准条目配对 | {single_missed} |", "",
              f"按全部标准条文标签取并集后的领域数量分布：{dict(sorted(coverage_counts.items()))}。",
              "保留率为 100% 是已知标准答案所属领域后的结构性检查；不证明模型能选对领域，也不证明未标注条文都不相关。",
              "《民法典》在本版整份挂入民事、婚姻和住房领域，因此这些领域的随机样例可能出现同一综合文件中其他主题的条文。这是文件级分类带入的噪声，不是条文级精细分类已经完成。", "",
              "## 8 个大类与 22 个领域", "", "跨领域重复计入各领域，大类按条文 ID 取并集；不能把各行直接相加当作总条文数。", ""]
    for parent_id, parent_name in PARENTS.items():
        parent_articles = set()
        parent_files = set()
        for domain in domains:
            if domain["parent_id"] == parent_id:
                parent_articles.update(domain_articles[domain["domain_id"]])
                parent_files.update(domain_files[domain["domain_id"]])
        report.extend([f"### {parent_id} {parent_name}", "", f"去重后：{len(parent_files)} 个文件组，{len(parent_articles):,} 条法条。", "",
                       "| 领域 | 文件组数 | 条文数 |", "| --- | ---: | ---: |"])
        for row in summary:
            if row["parent_id"] == parent_id:
                report.append(f"| {row['domain_id']} {row['domain_name']} | {row['file_count']} | {row['article_count']:,} |")
        report.append("")
    report.extend(["## 每领域随机 5 条样例", "", f"固定随机种子 {SEED}；每领域先随机选 5 份不同文件，再每份抽一条，便于检查文件分类。正文仅展示前 180 字，完整正文可按 ID 查 data/raw/STARD/corpus.jsonl。", ""])
    for domain in domains:
        report.extend([f"### {domain['domain_id']} {domain['name']}", ""])
        for row in samples:
            if row["domain_id"] == domain["domain_id"]:
                excerpt = " ".join(row["content"].replace("\\n", " ").split())[:180]
                report.extend([f"- **{row['name']}**（ID {row['article_id']}）", f"  {excerpt}"])
        report.append("")
    report.extend(["## 强制只选一个领域时会遗漏的题", "", "以下仍是看过标准答案后选出的最佳单领域，不是实际预测。下列案例及汇总可通过data/processed/classify_domains.py 重算。", ""])
    shown = 0
    for row in query_results:
        if row["single_missing_gold"] and shown < 10:
            names = []
            for article_id in row["single_missing_gold"]:
                names.append(by_id[article_id]["name"])
            report.append(f"- Q{row['query_id']}：{row['query']}；最佳单领域 {row['best_single_domain']} 仍遗漏：{'；'.join(names)}。")
            shown += 1
    report.extend(["", "## 复现与数据版本", "",
                   "在仓库根目录执行：`python data/processed/classify_domains.py`。结果写入 data/processed/file_domains.jsonl，说明写入 data/processed/数据集分类规则.md；不调用外部模型。",
                   f"随机种子：{SEED}。文件分类清单记录文件 ID、名称、所含法条 ID、领域 ID 和匹配依据。",
                   f"corpus.jsonl SHA-256：`{stats['corpus.jsonl_sha256']}`。",
                   f"queries.json SHA-256：`{stats['queries.json_sha256']}`。",
                   "原始数据来源：https://github.com/oneal2000/STARD 。"])
    (OUT / "数据集分类规则.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    assert len(corpus) == len(article_domains)
    assert len(samples) == 5 * len(domains)
    assert sum(coverage_counts.values()) == len(queries)
    assert stats["gold_informed_missing_gold_pairs"] == 0
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
