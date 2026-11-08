from scripts.stratify_intro_train import excluded_component, flags, rank


def row(title, text, page_id='1'):
    return {'title': title, 'text': text, 'page_id': page_id, 'group': page_id}


def test_subject_list_and_personal_information_are_quarantined():
    assert 'dependent_opening' in flags(row('某地', '是一个位于山中的村庄。'))
    assert 'subject_not_anchored' in flags(row('某地', '位于山中的村庄有古老的桥。'))
    assert 'list_residue' in flags(row('某地', '某地包括：甲、乙、丙。'))
    assert 'personal_sensitive_risk' in flags(row('某人', '某人的年薪据传为十万元。'))
    assert 'personal_sensitive_risk' in flags(row('某事件', '某事件涉及未成年人。'))


def test_normal_prose_can_pass_and_sensitive_terms_can_false_positive():
    assert flags(row('山茶', '山茶是一种植物，花朵在冬季开放。')) == []
    assert 'personal_sensitive_risk' in flags(row('财政', '财政收入是政府的重要资金来源。'))


def test_exclusion_uses_page_and_near_duplicate_component():
    base = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥' * 3
    rows = [row('甲', base, '1'), row('乙', base + '天地', '2'),
            row('丙', '完全不同的内容需要保留下来' * 4, '3')]
    prior = [{'id': '1', 'text': base}]
    assert excluded_component(rows, prior) == {0, 1}


def test_rank_is_repeatable_and_seeded():
    value = row('甲', '甲是一段文本。')
    assert rank('one', value) == rank('one', value)
    assert rank('one', value) != rank('two', value)
