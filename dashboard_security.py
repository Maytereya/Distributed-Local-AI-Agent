"""Server-side role checks for legacy Gradio admin callbacks."""
import contextvars
import functools
import inspect
from contextlib import contextmanager
import gradio as gr
from gradio.blocks import BlocksConfig

_ADMIN_SCOPE=contextvars.ContextVar("dashboard_admin_scope",default=False)
_BASIC_MODES={"Free-talk-Ai","Call-Center-Ai","Messengers-Ai"}


def _role(username):
    from agent_logic_2.gradio_ui.shared.auth import get_user_role
    return get_user_role(username)


def guarded(fn,*,admin=True,role_fn=None):
    role_fn=role_fn or _role
    signature=inspect.signature(fn)
    params=list(signature.parameters.values())
    # functools.partial turns pre-bound keyword options into keyword-only
    # parameters. Keep them fixed inside the original callable, outside the
    # browser input signature (e.g. Enable/Disable topic buttons).
    if isinstance(fn,functools.partial):
        params=[p for p in params if not (p.kind==p.KEYWORD_ONLY and p.name in (fn.keywords or {}))]
        signature=signature.replace(parameters=params)
    if any(p.kind not in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD) for p in params):
        raise RuntimeError("unsupported_dashboard_callback_signature")
    index=len(params)
    params.append(inspect.Parameter("_security_request",inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    default=None,annotation=gr.Request))

    def checked(args,kwargs):
        args=list(args)
        request=kwargs.pop("_security_request",None)
        if len(args)>index: request=args.pop(index)
        username=getattr(request,"username",None)
        role=role_fn(username) if username else None
        if not username or (admin and role!="admin"):
            raise gr.Error("Недостаточно прав для этого действия.",print_exception=False)
        if not admin and role!="admin":
            bound=signature.bind_partial(*args,**kwargs)
            if bound.arguments.get("radio_value") not in _BASIC_MODES:
                raise gr.Error("Этот режим доступен только администратору.",print_exception=False)
        return args,kwargs

    if inspect.isasyncgenfunction(fn):
        async def wrapper(*args,**kwargs):
            args,kwargs=checked(args,kwargs)
            async for item in fn(*args,**kwargs): yield item
    elif inspect.iscoroutinefunction(fn):
        async def wrapper(*args,**kwargs):
            args,kwargs=checked(args,kwargs)
            return await fn(*args,**kwargs)
    elif inspect.isgeneratorfunction(fn):
        def wrapper(*args,**kwargs):
            args,kwargs=checked(args,kwargs)
            yield from fn(*args,**kwargs)
    else:
        def wrapper(*args,**kwargs):
            args,kwargs=checked(args,kwargs)
            return fn(*args,**kwargs)
    functools.update_wrapper(wrapper,fn)
    wrapper.__signature__=signature.replace(parameters=params)
    wrapper.__annotations__={**getattr(fn,"__annotations__",{}),"_security_request":gr.Request}
    wrapper._dashboard_guarded=True
    return wrapper


@contextmanager
def admin_callbacks():
    token=_ADMIN_SCOPE.set(True)
    try: yield
    finally: _ADMIN_SCOPE.reset(token)


def install_dashboard_policy():
    original=BlocksConfig.set_event_trigger
    if getattr(original,"_dashboard_guarded",False): return
    signature=inspect.signature(original)
    @functools.wraps(original)
    def register(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        fn=bound.arguments.get("fn")
        if fn and not getattr(fn,"_dashboard_guarded",False):
            if _ADMIN_SCOPE.get(): bound.arguments["fn"]=guarded(fn)
            elif getattr(fn,"__name__","")=="universal_echo": bound.arguments["fn"]=guarded(fn,admin=False)
        return original(*bound.args,**bound.kwargs)
    register._dashboard_guarded=True
    BlocksConfig.set_event_trigger=register
