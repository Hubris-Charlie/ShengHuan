"""关单方式判定的规则测试。

这几条规则直接来自《资审关单场景话术》主表的「关单总原则」，
判错会导致"该打的没打"或"不该打的乱打"，属于业务红线，必须锁死。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.policy import (
    CallContext,
    CloseMethod,
    HOTLINE,
    NextAction,
    Scenario,
    decide_call,
    find_scenario,
    get_scenario,
    load_scenarios,
    phone_scenarios,
)

PHONE_SCENE = Scenario("S001", "品牌", "商标注册证", "商标状态不符合要求", CloseMethod.PHONE, "话术")
JM_SCENE = Scenario("S003", "品牌", "商标注册证", "资质造假", CloseMethod.JINGMAI, "话术")


class TestRule3MerchantRequested(unittest.TestCase):
    def test_jingmai_scene_still_calls(self):
        """商家主动要求回电 → 即使场景是京麦驳回也必须打。"""
        d = decide_call(JM_SCENE, CallContext(merchant_requested_callback=True))
        self.assertTrue(d.should_call)
        self.assertIs(d.method, CloseMethod.PHONE)
        self.assertEqual(d.reason_code, "merchant_requested")


class TestRule4Escalation(unittest.TestCase):
    def test_escalation_overrides_everything(self):
        """越级投诉/外部曝光 → 一律回电，且优先级高于所有其他规则。"""
        for scene in (PHONE_SCENE, JM_SCENE):
            for ctx in (
                CallContext(escalation_risk=True),
                CallContext(escalation_risk=True, attempt_count=5),
                CallContext(escalation_risk=True, already_called_unreached=True, attempt_count=2),
            ):
                d = decide_call(scene, ctx)
                self.assertTrue(d.should_call, f"{scene.id} {ctx} 应回电")
                self.assertTrue(d.need_escalation)
                self.assertIs(d.action, NextAction.OUTBOUND_ESCALATED)

    def test_escalation_beats_third_attempt(self):
        d = decide_call(PHONE_SCENE, CallContext(attempt_count=3, escalation_risk=True))
        self.assertTrue(d.should_call)


class TestRule2Attempts(unittest.TestCase):
    def test_first_attempt_normal_call(self):
        d = decide_call(PHONE_SCENE, CallContext(attempt_count=1))
        self.assertTrue(d.should_call)
        self.assertIs(d.action, NextAction.OUTBOUND)

    def test_first_unreached_falls_back_to_jingmai(self):
        """电话场景首次未接通 → 京麦驳回告知原因即可。"""
        d = decide_call(PHONE_SCENE, CallContext(attempt_count=1, already_called_unreached=True))
        self.assertFalse(d.should_call)
        self.assertIs(d.method, CloseMethod.JINGMAI)
        self.assertEqual(d.reason_code, "first_unreached")

    def test_second_unreached_schedules_second_call(self):
        """二次提交仍错 + 上次未接通 → 创事件预约二次外呼，不是直接京麦。"""
        d = decide_call(PHONE_SCENE, CallContext(attempt_count=2, already_called_unreached=True))
        self.assertTrue(d.should_call)
        self.assertIs(d.action, NextAction.SCHEDULE_SECOND_CALL)
        self.assertEqual(d.reason_code, "second_attempt_schedule")

    def test_third_attempt_always_jingmai(self):
        for ctx in (
            CallContext(attempt_count=3),
            CallContext(attempt_count=3, already_called_unreached=True),
            CallContext(attempt_count=9),
        ):
            d = decide_call(PHONE_SCENE, ctx)
            self.assertFalse(d.should_call, f"{ctx} 三次及以上应京麦驳回")
            self.assertIs(d.method, CloseMethod.JINGMAI)


class TestJingmaiScene(unittest.TestCase):
    def test_jingmai_not_forced_to_call(self):
        d = decide_call(JM_SCENE, CallContext())
        self.assertFalse(d.should_call)
        self.assertIs(d.method, CloseMethod.JINGMAI)
        self.assertEqual(d.reason_code, "jingmai_scene")


class TestScenarioLibrary(unittest.TestCase):
    def test_loaded_counts(self):
        s = load_scenarios()
        self.assertEqual(len(s), 68, "主表应有 68 条场景")
        phone = phone_scenarios()
        self.assertGreater(len(phone), 0)
        # 分类树已向下填充，不会出现空的一级菜单
        for sc in s.values():
            self.assertTrue(sc.menu1, f"{sc.id} 一级菜单为空，向下填充没生效")

    def test_menu1_distribution(self):
        from collections import Counter

        c = Counter(s.menu1 for s in load_scenarios().values())
        self.assertEqual(c["品牌"], 33)
        self.assertEqual(c["入驻"], 25)
        self.assertEqual(c["行业资质"], 4)
        self.assertEqual(c["公司资质"], 4)
        self.assertEqual(c["商品资质"], 2)

    def test_find_by_tree(self):
        sc = find_scenario("品牌", "商标注册证", "商标状态不符合要求")
        self.assertIsNotNone(sc)
        self.assertIs(sc.close_method, CloseMethod.PHONE)
        self.assertIn(HOTLINE, sc.script)

    def test_get_by_id(self):
        sc = get_scenario("S001")
        self.assertIsNotNone(sc)
        self.assertEqual(sc.id, "S001")

    def test_jingmai_scene_example(self):
        sc = find_scenario("品牌", "商标注册证", "资质造假（公章造假/链路造假等）")
        self.assertIsNotNone(sc)
        self.assertIs(sc.close_method, CloseMethod.JINGMAI)


if __name__ == "__main__":
    unittest.main(verbosity=2)
