import os
os.environ["GRADIO_ANALYTICS_ENABLED"]="False"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
import unittest
from functools import partial
from unittest.mock import Mock
import gradio as gr
from gradio.helpers import special_args
from dashboard_security import guarded,admin_callbacks,install_dashboard_policy


class DashboardSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_prebound_admin_button_option_cannot_be_overridden(self):
        calls=[]
        def toggle(topic,enabled): calls.append((topic,enabled))
        fn=guarded(partial(toggle,enabled=True),role_fn=lambda _:"admin")
        args,_,_=special_args(fn,["synthetic_topic"],request=gr.Request(username="owner"))
        fn(*args)
        self.assertEqual(calls,[("synthetic_topic",True)])
    async def test_gradio_injects_authenticated_request_and_denies_before_mutation(self):
        operation=Mock(return_value="safe_result")
        def change(value): return operation(value)
        fn=guarded(change,role_fn=lambda name:"admin" if name=="owner" else "basic")
        for username in (None,"ordinary"):
            args,_,_=special_args(fn,["PATIENT_CANARY"],request=gr.Request(username=username))
            with self.assertRaises(gr.Error) as error: fn(*args)
            self.assertNotIn("PATIENT_CANARY",str(error.exception))
        operation.assert_not_called()
        args,_,_=special_args(fn,["allowed"],request=gr.Request(username="owner"))
        self.assertEqual(fn(*args),"safe_result")
        operation.assert_called_once_with("allowed")

    async def test_async_stream_modes_cannot_bypass_hidden_radio_choices(self):
        calls=[]
        async def universal_echo(message,radio_value):
            calls.append(radio_value)
            yield "safe_result"
        fn=guarded(universal_echo,admin=False,role_fn=lambda _:"basic")
        for mode in ("db","gigachat"):
            args,_,_=special_args(fn,["PATIENT_CANARY",mode],request=gr.Request(username="ordinary"))
            with self.assertRaises(gr.Error): [value async for value in fn(*args)]
        self.assertEqual(calls,[])
        args,_,_=special_args(fn,["allowed","Call-Center-Ai"],request=gr.Request(username="ordinary"))
        self.assertEqual([value async for value in fn(*args)],["safe_result"])

    async def test_admin_tab_registration_keeps_role_check_on_actual_gradio_callback(self):
        install_dashboard_policy()
        def read(value): return value
        with gr.Blocks(analytics_enabled=False) as app:
            field=gr.Textbox()
            button=gr.Button()
            with admin_callbacks(): button.click(read,inputs=field,outputs=field)
        self.assertTrue(app.fns[0].fn._dashboard_guarded)
        args,_,_=special_args(app.fns[0].fn,["PATIENT_CANARY"],request=gr.Request(username=None))
        with self.assertRaises(gr.Error): app.fns[0].fn(*args)
