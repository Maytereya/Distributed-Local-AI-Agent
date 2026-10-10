import unittest
from reporting_projection import EVENT_LABELS, failure_card, maintenance_priority


def case(number=41,**fields):
    return {"id":number,"outcome":"H","cause":"requested_operator","action":"none","status":"typed_evidence","confidence":None,
        "tasks":{"PRICE":{"state":"pending","kind":"none"}},"events":{k:0 for k in EVENT_LABELS},**fields}


class FailureCardsTests(unittest.TestCase):
    def projection(self,cases,total=None):
        return {"version":1,"enabled":True,"quality_passed":False,"cases":cases,"total":total or len(cases),"truncated":total is not None and total>len(cases)}

    def test_operator_choice_is_not_reported_as_confirmed_bot_defect(self):
        text=failure_card(self.projection([case()]),detail=True)
        self.assertIn("Ошибка бота не установлена",text)
        self.assertIn("http://172.16.0.28:8080/reporting/appeals/41/",text)
        self.assertIn("вход действующим оператором через VPN",text)
        self.assertIn("качество ИИ-анализатора ещё не подтверждено",text)

    def test_summary_keeps_partial_separate_and_limits_counts_to_shown_subset(self):
        first=case(outcome="P",cause="partial_answer",action="check_scenario")
        first["events"]["delivery_uncertain"]=1
        text=failure_card(self.projection([first,case(42)],total=120))
        self.assertIn("Обращений в выборке: 120",text)
        self.assertIn("В показанной выборке обращений: 2",text)
        self.assertIn("Неуспешных или с неизвестным итогом: 1",text)
        self.assertIn("Частично решённых: 1",text)
        self.assertIn("Доставка не подтверждена: 1",text)
        self.assertIn("Категории могут пересекаться",text)

    def test_maintenance_priority_cannot_demote_urgent_scenario(self):
        self.assertTrue(maintenance_priority({"URGENT:1":{}},"requested_operator").startswith("Высокий"))
        self.assertTrue(maintenance_priority({},"technical_error").startswith("Высокий"))
        self.assertIn("Требуется ручная оценка",maintenance_priority({},"unknown"))
