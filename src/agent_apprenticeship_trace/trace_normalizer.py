from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from .schemas import AgentTrace, AgentTraceStep, ActualOutputs
from .io import write_json
from .artifact_resolver import normalize_artifact_ref

CANONICAL_TOP = {'schema_version','trace_id','collection_id','run_id','package_id','bundle_id','prior_trace_id','trace_mode','task','task_id','task_family_id','attempt_id','attempt_kind','attempt_status','agent_tools','started_at','ended_at','system_prompt','system_prompt_hash','skills','memory','agent_config','environment_domain','actual_outputs_ref','input_artifact_refs','artifact_refs','deliverable_refs','output_summary','final_output_summary','learning','termination_reason','steps','actual_outputs','artifacts','metadata_json'}
CANONICAL_STEP = set(AgentTraceStep.model_fields)
VALID_OUTCOMES={'progress','neutral','blocked','failed','corrected','completed'}
VALID_ACTIONS={'user_message','agent_step','output','error'}
VALID_OPS={'plan','analyze','search','read','write','edit','execute','verify','download','install','ask_user','answer','select','grade','evaluate','revise','other'}
VALID_CAUSAL_TYPES={'user_request','follow_up_user_request','answer_to_agent_question','execution_of_plan','dependency_on_tool_result','retry_after_failure','correction_response','approval_response','verification_of_prior_step','dependency_on_multiple_prior_steps','delegation_to_subagent','delegated_work','used_subagent_result','handoff_from_subagent','parallel_work','other'}
CAUSAL_MAP={'sequential':'execution_of_plan','sequence':'execution_of_plan','previous_step':'execution_of_plan'}
OP_MAP={
 'inspect_workspace_files':'read','inspect_workspace':'read','read_policy':'read','read_input_csvs':'read','create_artifacts_directory':'execute','create_reconciliation_script':'write','run_initial_reconciliation':'execute','inspect_initial_outputs':'read','generate_reconciliation_artifacts':'execute','inspect_reconciled_payments':'read','validate_json_files':'verify','validate_artifacts':'verify','run':'execute','shell':'execute','bash':'execute','create_file':'write','write_file':'write','modify_file':'edit','read_file':'read','inspect':'read','think':'analyze','reason':'analyze','summarize':'analyze','finalize':'answer','final_response':'answer'}
READ_OPS={'read','search'}; WRITE_OPS={'write','edit'}; EXEC_OPS={'execute','install','download'}; VERIFY_OPS={'verify'}
VALID_ATTEMPT_KINDS={'baseline','revised','apprentice_without_lessons','apprentice_with_lessons','other'}
VALID_TRACE_MODES={'live','retraced','hybrid'}
VALID_TERMINATION_REASONS={'task_complete','verifier_passed','verifier_failed','max_iterations_reached','agent_blocked','timeout','error_unrecoverable','partial_then_stopped','provider_usage_limit','other'}
VALID_ATTEMPT_STATUSES={'completed','failed','blocked','fallback','partial'}
ENVIRONMENT_DOMAINS={'terminal','browser','computer_use','web','search','codebase','software','os','android','mcp','api','database','spreadsheet','file','multimodal','device','robotics','lab','industrial','physical','mixed','unknown'}
INTERACTION_SURFACES={'api','mcp_tool','mcp_resource','mcp_prompt','browser','computer_use','cli','file','codebase','database','spreadsheet','search','cloud_console','saas_app','async_job','queue','webhook','scheduler','messaging','email','collaboration','document','pdf','table','multimodal','device','industrial','physical','mixed','other_tool','other_surface','unknown'}
BASE_STEP_FIELDS={'step','turn','actor','action','operation','tool','environment_domain','execution_mode','parallel_group','observation','input','input_source','output','state_change','reasoning','caused_by','causal_type','causal_note','alternatives_considered','success','step_outcome','error_type','error_message','message_role','feedback_type','feedback_content','started_at','ended_at','retry_of','artifact_refs','metadata_json'}
SURFACE_STEP_FIELDS=CANONICAL_STEP-BASE_STEP_FIELDS
LIST_SURFACE_FIELDS={name for name in SURFACE_STEP_FIELDS if name.endswith('_refs') or name in {'modalities','connected_surfaces','depends_on_agent_outputs','formula_refs','pivot_refs','chart_refs','data_validation_refs','file_paths','selected_result_refs','citation_refs','attachment_refs','decision_refs','page_refs','section_refs','table_extract_refs','figure_refs','annotation_refs','safety_limit_refs','compliance_refs','cross_system_dependencies','handoff_refs','state_sync_refs','consistency_checks','evidence_refs','verification_refs','redaction_notes','missing_surface_fields'}}
INT_SURFACE_FIELDS={'http_status','retry_count','row_count','affected_row_count','column_count','result_count','timeout_seconds'}
FLOAT_SURFACE_FIELDS={'latency_ms'}
BOOL_SURFACE_FIELDS={'operator_override','human_supervisor_present','emergency_stop','followup_required'}

class TraceNormalizationContext(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    task: str
    actual_outputs: ActualOutputs | None = None
    trace_id_prefix: str = 'trace'

class TraceNormalizationReport(BaseModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    raw_trace_ref: str | None = None
    normalized_trace_ref: str | None = None
    canonical_trace_ref: str | None = None
    trace_schema_valid: bool = False
    trace_normalized: bool = False
    trace_lossless: bool = True
    fallback_trace: bool = False
    raw_step_count: int = 0
    normalized_step_count: int = 0
    discarded_step_count: int = 0
    raw_trace_parse_error: bool = False
    trace_normalization_error: bool = False
    trace_normalization_partial: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    field_mappings: dict[str,str] = Field(default_factory=dict)
    input_dict_to_string_count: int = 0
    output_dict_to_string_count: int = 0
    causal_type_repair_count: int = 0
    operation_repair_count: int = 0
    action_repair_count: int = 0
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class NormalizationResult(BaseModel):
    normalized_trace: dict[str, Any] | None = None
    report: TraceNormalizationReport
    fallback_required: bool = False
    parse_error: str | None = None

def _compact(v: Any, max_len: int=1200) -> str:
    if v is None: return ''
    if isinstance(v, str): return v[:max_len]
    try: s=json.dumps(v, sort_keys=True)
    except Exception: s=str(v)
    return s[:max_len]

def _as_list(v: Any) -> list[Any]:
    if v is None: return []
    if isinstance(v, list): return v
    return [v]

def _normalize_ref(ref: Any) -> str:
    value = normalize_artifact_ref(ref).replace("\\", "/").strip().lstrip("/")
    while value.startswith("./"):
        value = value[2:]
    return value

def _normalize_ref_list(value: Any) -> list[str]:
    refs: list[str] = []
    for item in _as_list(value):
        ref = _normalize_ref(item)
        if ref and ref not in refs:
            refs.append(ref)
    return refs

def _actual_ref(attempt_kind: str) -> str:
    return f"attempts/{attempt_kind}/actual_outputs.json"

def _input_source(value: Any, meta: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return {"source": value.strip()}
    if value is not None:
        meta["original_input_source"] = value
    return None

def _actor_to_string(value: Any, action: str, meta: dict[str, Any]) -> str:
    if action == "user_message":
        return "user"
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        meta["original_actor"] = value
        actor_kind = str(value.get("actor_kind") or value.get("kind") or "").strip()
        role = str(value.get("role") or value.get("actor_role") or value.get("name") or "").strip()
        agent_id = str(value.get("agent_id") or value.get("id") or "").strip()
        if actor_kind and role:
            return f"{actor_kind}:{role}"
        if actor_kind and agent_id:
            return f"{actor_kind}:{agent_id}"
        if actor_kind:
            return actor_kind
        if role:
            return role
        if agent_id:
            return f"agent:{agent_id}"
    elif value is not None:
        meta["original_actor"] = value
    return "agent:worker"

def _infer_environment_domain(tool: Any, operation: Any = None, input_value: Any = None, output_value: Any = None) -> str:
    text = " ".join(str(x or "") for x in [tool, operation, input_value, output_value]).lower()
    if any(x in text for x in ["mcp__", "mcp ", "model context protocol"]):
        return "mcp"
    if any(x in text for x in ["robot", "robotics", "actuator", "telemetry", "sensor", "calibration", "emergency stop"]):
        return "robotics" if "robot" in text else "physical"
    if any(x in text for x in ["industrial", "manufacturing", "plc", "scada", "field device", "lab instrument"]):
        return "industrial"
    if any(x in text for x in ["browser", "playwright", "screenshot", "dom", "page.goto"]):
        return "browser"
    if any(x in text for x in ["computer_use", "desktop app", "ui automation", "accessibility tree"]):
        return "computer_use"
    if any(x in text for x in ["http://", "https://", "fetch", "curl ", "web "]):
        return "web"
    if any(x in text for x in ["search", "web_search"]):
        return "search"
    if any(x in text for x in ["sqlite", "postgres", "mysql", "sql ", "database", "db "]):
        return "database"
    if any(x in text for x in ["xlsx", "xls", "spreadsheet", "workbook", "csv"]):
        return "spreadsheet"
    if any(x in text for x in ["pdf", "document", "ocr", "transcript", "image", "video", "audio", "screenshot"]):
        return "multimodal"
    if any(x in text for x in ["android", "adb "]):
        return "android"
    if any(x in text for x in ["api", "json endpoint", "graphql", "rest "]):
        return "api"
    if any(x in text for x in ["apply_patch", "pytest", "compileall", "git ", "npm ", "node ", "python", "rg ", "grep ", "codebase"]):
        return "codebase"
    if any(x in text for x in ["bash", "zsh", "shell", "terminal", "subprocess", "command"]):
        return "terminal"
    if any(x in text for x in ["file_read", "file_write", "read_file", "write_file", "input/", "artifacts/"]):
        return "file"
    if any(x in text for x in ["os.", "filesystem", "mkdir", "cp ", "mv ", "rm "]):
        return "os"
    return "unknown"


def _infer_interaction_surface(raw: dict[str, Any], tool: Any, operation: Any, input_value: Any, output_value: Any) -> str:
    explicit = raw.get("interaction_surface")
    if explicit in INTERACTION_SURFACES:
        return explicit
    text = " ".join(str(x or "") for x in [tool, operation, input_value, output_value, raw.get("surface_type"), raw.get("app_name")]).lower()
    if any(x in text for x in ["mcp__", "mcp tool", "mcp_tool"]):
        return "mcp_tool"
    if any(x in text for x in ["mcp resource", "mcp_resource"]):
        return "mcp_resource"
    if any(x in text for x in ["mcp prompt", "mcp_prompt"]):
        return "mcp_prompt"
    if any(x in text for x in ["curl ", "http ", "https://", "graphql", "rest api", "sdk", "api "]):
        return "api"
    if any(x in text for x in ["sql", "postgres", "mysql", "sqlite", "database", "db "]):
        return "database"
    if any(x in text for x in ["xlsx", "workbook", "spreadsheet", "sheet_name"]):
        return "spreadsheet"
    if any(x in text for x in ["csv", "table", "dataframe"]):
        return "table"
    if any(x in text for x in ["browser", "playwright", "dom", "page.", "url"]):
        return "browser"
    if any(x in text for x in ["desktop app", "computer_use", "accessibility tree", "screenshot"]):
        return "computer_use"
    if any(x in text for x in ["web_search", "search", "query"]):
        return "search"
    if any(x in text for x in ["cloud console", "aws console", "gcp console", "azure portal"]):
        return "cloud_console"
    if any(x in text for x in ["saas", "crm", "ticket", "tenant"]):
        return "saas_app"
    if any(x in text for x in ["webhook", "callback"]):
        return "webhook"
    if any(x in text for x in ["queue", "message_id", "dead letter"]):
        return "queue"
    if any(x in text for x in ["cron", "scheduler", "schedule"]):
        return "scheduler"
    if any(x in text for x in ["async job", "job_id", "polling"]):
        return "async_job"
    if any(x in text for x in ["email", "smtp", "inbox"]):
        return "email"
    if any(x in text for x in ["slack", "teams", "discord", "message", "thread"]):
        return "messaging"
    if any(x in text for x in ["comment", "collaboration", "shared doc"]):
        return "collaboration"
    if "pdf" in text:
        return "pdf"
    if any(x in text for x in ["document", "docx", "ocr", "report"]):
        return "document"
    if any(x in text for x in ["image", "video", "audio", "multimodal"]):
        return "multimodal"
    if any(x in text for x in ["robot", "industrial", "sensor", "actuator", "telemetry", "device"]):
        return "industrial" if "industrial" in text else "device"
    if any(x in text for x in ["apply_patch", "git ", "pytest", "npm ", "python", "node ", "rg ", "bash", "shell", "command"]):
        return "cli"
    if any(x in text for x in ["file_read", "file_write", "read_file", "write_file", "artifact", "input/"]):
        return "file"
    if any(x in text for x in ["codebase", "test", "build", "lint", "compile"]):
        return "codebase"
    return "unknown"


def _surface_required_fields(surface: str) -> list[str]:
    return {
        "api": ["api_endpoint", "api_method", "http_status"],
        "mcp_tool": ["mcp_server", "mcp_tool_name", "mcp_result_ref"],
        "mcp_resource": ["mcp_server", "mcp_resource_uri", "mcp_result_ref"],
        "mcp_prompt": ["mcp_server", "mcp_prompt_name", "mcp_result_ref"],
        "database": ["database_type", "query_ref", "table_name"],
        "spreadsheet": ["workbook_ref", "sheet_name", "cell_range"],
        "table": ["table_ref", "row_count", "column_count"],
        "cli": ["command", "exit_code", "stdout_ref"],
        "file": ["file_path", "file_operation"],
        "codebase": ["command", "test_result_ref", "diff_ref"],
        "browser": ["visible_state_ref", "screenshot_ref", "element_selector", "action_type"],
        "computer_use": ["visible_state_ref", "screenshot_ref", "action_type"],
        "async_job": ["job_id", "job_status_after", "async_result_ref"],
        "queue": ["queue_name", "message_id", "delivery_status"],
        "webhook": ["webhook_id", "callback_status", "async_result_ref"],
        "scheduler": ["schedule_id", "trigger_type", "job_status_after"],
        "device": ["device_type", "telemetry_ref", "safety_state"],
        "industrial": ["physical_environment_type", "telemetry_ref", "safety_state"],
        "physical": ["physical_environment_type", "physical_state_after_ref", "safety_state"],
    }.get(surface, [])


def _normalize_surface_value(field: str, value: Any, meta: dict[str, Any]) -> Any:
    if field in LIST_SURFACE_FIELDS:
        if field.endswith("_refs") or field in {"file_paths", "selected_result_refs", "citation_refs", "attachment_refs", "decision_refs", "page_refs", "section_refs", "table_extract_refs", "figure_refs", "annotation_refs", "safety_limit_refs", "compliance_refs", "handoff_refs", "state_sync_refs", "evidence_refs", "verification_refs"}:
            return _normalize_ref_list(value)
        return [str(v) for v in _as_list(value) if v not in (None, "")]
    if field.endswith("_ref"):
        return _normalize_ref(value)
    if field in INT_SURFACE_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError):
            meta.setdefault("surface_field_parse_warnings", []).append(f"{field}_not_integer")
            return None
    if field in FLOAT_SURFACE_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            meta.setdefault("surface_field_parse_warnings", []).append(f"{field}_not_float")
            return None
    if field in BOOL_SURFACE_FIELDS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return value.lower() in {"true", "yes", "1"}
        return None
    return _string_or_none(value, meta, field)


def _surface_completeness(step: dict[str, Any], meta: dict[str, Any]) -> tuple[str, list[str]]:
    surface = step.get("interaction_surface")
    if not surface or surface in {"unknown", "other_tool", "other_surface"}:
        return "not_applicable", []
    required = _surface_required_fields(surface)
    missing = [name for name in required if step.get(name) in (None, "", [], {})]
    if surface == "cli":
        if "command" in missing and step.get("input"):
            missing.remove("command")
        if "stdout_ref" in missing and step.get("output"):
            meta.setdefault("surface_evidence_substitutions", []).append("cli_stdout_ref_satisfied_by_output")
            missing.remove("stdout_ref")
        if "exit_code" in missing and step.get("success") is not None:
            meta.setdefault("surface_evidence_substitutions", []).append("cli_exit_code_satisfied_by_success_signal")
            missing.remove("exit_code")
    if surface == "file":
        if "file_path" in missing and (step.get("artifact_refs") or step.get("input")):
            missing.remove("file_path")
        if "file_operation" in missing and step.get("operation"):
            missing.remove("file_operation")
    if surface == "codebase":
        if "command" in missing and step.get("input"):
            missing.remove("command")
        if "test_result_ref" in missing and step.get("output"):
            meta.setdefault("surface_evidence_substitutions", []).append("codebase_test_result_ref_satisfied_by_output")
            missing.remove("test_result_ref")
        if "diff_ref" in missing and (step.get("artifact_refs") or step.get("state_change")):
            meta.setdefault("surface_evidence_substitutions", []).append("codebase_diff_ref_satisfied_by_artifact_or_state_change")
            missing.remove("diff_ref")
    if not required:
        return "partial" if step.get("output") or step.get("state_change") or step.get("artifact_refs") else "unavailable", []
    if not missing:
        return "complete", []
    if step.get("surface_capture_notes") or meta.get("surface_capture_unavailable_reason"):
        return "unavailable", missing
    return "partial", missing

def _operation(raw_action: str | None, raw_op: str | None, meta: dict[str,Any], report: TraceNormalizationReport | None = None) -> str:
    val=str(raw_op or raw_action or '').strip()
    mapped=OP_MAP.get(val, val if val in VALID_OPS else 'other')
    if val and (val != mapped or val not in VALID_OPS):
        meta['original_operation']=val
        if report is not None:
            report.operation_repair_count += 1
    return mapped if mapped in VALID_OPS else 'other'

def _action(raw_action: str | None, op: str, meta: dict[str,Any], report: TraceNormalizationReport | None = None) -> str:
    a=str(raw_action or '').strip()
    low=a.lower()
    if a:
        meta['original_action']=a
    if low in VALID_ACTIONS:
        return low
    if low in {'error','failure','exception'} or (op == 'other' and any(x in low for x in ['error','fail','exception'])):
        canonical='error'
    elif low in {'final_response','finalize'} or op == 'answer':
        canonical='output'
    else:
        canonical='agent_step'
    if a and report is not None:
        report.action_repair_count += 1
    return canonical

def _string_or_none(v: Any, meta: dict[str, Any] | None = None, key: str | None = None) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if meta is not None and key is not None:
        meta[key]=v
    return _compact(v)

def _output_to_string(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        parts=[]
        for k in ['exit_code','stdout','stdout_summary','stderr','stderr_summary']:
            if k in v and v[k] not in (None, ''):
                parts.append(f'{k}: {_compact(v[k], 900)}')
        return '\n'.join(parts) if parts else _compact(v)
    return _compact(v)

def _normalize_causal_type(v: Any, meta: dict[str, Any], report: TraceNormalizationReport | None = None) -> str | None:
    if v is None:
        return None
    raw=str(v).strip()
    mapped=CAUSAL_MAP.get(raw, raw if raw in VALID_CAUSAL_TYPES else 'other')
    if raw != mapped or raw not in VALID_CAUSAL_TYPES:
        meta['original_causal_type']=v
        if report is not None:
            report.causal_type_repair_count += 1
    return mapped


def _command_tokens(command: str) -> list[str]:
    return [part.strip().lower() for part in command.replace('&&', '\n').replace(';', '\n').split('\n') if part.strip()]

def _first_command_word(part: str) -> str:
    words=part.split()
    while words and ('=' in words[0] or words[0] in {'env','time','uv','poetry','pipenv','xargs','sudo'}):
        words=words[1:]
    return words[0] if words else ''



def _has_phrase(text: str | None, phrase: str) -> bool:
    """Return true when a normalized phrase appears in text.

    Multi-word phrases use substring matching. Single-word phrases require
    token/word boundaries so generic language like "next move" does not
    accidentally become a file-write operation.
    """
    import re

    if not text or not phrase:
        return False

    text_l = str(text).lower()
    phrase_l = str(phrase).lower().strip()
    if not phrase_l:
        return False

    if " " in phrase_l:
        return phrase_l in text_l

    return re.search(r"(?<![a-z0-9_])" + re.escape(phrase_l) + r"(?![a-z0-9_])", text_l) is not None


def _operation_from_phrase(value: str | None) -> str | None:
    """Infer canonical operation from descriptive action/operation/tool text.

    Canonical operations are intentionally small:
    read, write, execute, verify, other.
    Return None when intent is unclear so command-level inference can still run.
    """
    if not value:
        return None

    low = str(value).lower().replace("_", " ").replace("-", " ").strip()
    if not low or low == "other":
        return None

    # Avoid mapping generic reasoning language to file operations.
    if "next move" in low or low.startswith(("ponder ", "think ", "reason ")):
        return None

    if low in {"read", "write", "execute", "verify"}:
        return low

    verify_markers = (
        "validate", "validation", "verify", "verification", "check", "test",
        "pytest", "compileall", "json tool", "schema", "checksum", "hash",
        "inspect exported", "inspect workbook", "formula scan", "scan workbook",
        "render", "visual qa", "preview", "qa", "review output",
        "spreadsheet validation", "xlsx validation", "csv validation",
        "markdown validation", "artifact validation",
        "attempted spreadsheet library import", "checked local office",
        "package availability", "tool check",
    )
    if any(_has_phrase(low, marker) for marker in verify_markers):
        return "verify"

    write_markers = (
        "write", "create", "edit", "patch", "apply patch", "file creation",
        "file edit", "write artifact", "create artifact", "edit artifact",
        "mkdir", "copy", "cp ",  "mv ", "builder script",
        "write final trace", "persist", "save", "generated artifacts",
    )
    if any(_has_phrase(low, marker) for marker in write_markers):
        return "write"

    execute_markers = (
        "execute", "run", "rerun", "re run", "executed", "script execution",
        "builder", "python", "python3", "node", "npm", "bash", "sh ",
        "calculation script", "link workspace", "environment setup",
        "symlink", "ln -s",
    )
    if any(_has_phrase(low, marker) for marker in execute_markers):
        return "execute"

    read_markers = (
        "read", "inspect", "inventory", "list", ("list" + "ing"), ("file " + "list" + "ing"),
        "file inspection", "read context", "inspect task", "inspect inputs",
        "inspect workspace", "load", "loaded", "sed", "cat", "head", "tail",
        "less", "ls", "find", "rg", "grep", "git status", "unzip",
    )
    if any(_has_phrase(low, marker) for marker in read_markers):
        return "read"

    return None


def _operation_from_command(command: str | None, meta: dict[str, Any], report: TraceNormalizationReport | None = None) -> str | None:
    c=(command or '').strip()
    if not c:
        return None
    parts=_command_tokens(c) or [c.lower()]
    low='\n'.join(parts)
    firsts={_first_command_word(p) for p in parts}
    verify_markers=['json.tool','pytest','compileall','schema','validate','validation','validator',' check','check ','checksum','sha256sum','shasum','md5sum','openpyxl','xlsx validation','csv validation','json validation','markdown validation','test ']
    write_markers=['apply_patch','file edit','edit file','write artifact','write_artifact','create file','file creation','write','create','edit','patch','builder script','write final trace','persist','save']
    execute_markers=['execute','run','rerun','re run','executed','script execution','builder','python','python3','node','npm','bash','sh','calculation script','environment setup','symlink','ln -s']
    read_markers=['read','inspect','inventory','list',('list' + 'ing'),('file ' + 'list' + 'ing'),'file inspection','read context','inspect task','inspect inputs','inspect workspace','load','loaded','sed','cat','head','tail','less','ls','find','rg','grep','git status','unzip']
    if any(m in low for m in verify_markers) or any(f in {'pytest','py.test'} for f in firsts):
        return 'verify'
    if 'git status' in low:
        return 'read'
    if any(m in low for m in ['apply_patch','file edit','edit file','write artifact','write_artifact','create file','file creation']):
        return 'write'
    if any(f in {'sed','cat','head','tail','less','ls','find','rg','grep','file','pwd'} for f in firsts):
        return 'read'
    if any('unzip -l' in p or 'zipinfo' in p for p in parts) or ' inventory' in low or ('list' + 'ing') in low:
        return 'read'
    if any(f in {'mkdir','cp','mv','touch','tee'} for f in firsts):
        return 'write'
    if any(f in {'python','python3','node','npm','npx','bash','sh','zsh'} for f in firsts):
        return 'execute'
    if any(_has_phrase(low, m) for m in verify_markers):
        return 'verify'
    if any(_has_phrase(low, m) for m in write_markers):
        return 'write'
    if any(_has_phrase(low, m) for m in read_markers):
        return 'read'
    return None

def _operation_from_step_context(raw_action: Any, raw_op: Any, tool: Any, command: Any, meta: dict[str, Any], report: TraceNormalizationReport | None = None) -> str | None:
    values=[raw_op, raw_action, tool, command]
    for value in values:
        op=_operation_from_phrase(str(value) if value is not None else None)
        if op:
            if report is not None:
                report.operation_repair_count += 1
            return op
    return None

def infer_attempt_status(trace: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    existing=trace.get('attempt_status')
    if existing in VALID_ATTEMPT_STATUSES:
        return existing
    termination=trace.get('termination_reason')
    md=trace.get('metadata_json') if isinstance(trace.get('metadata_json'), dict) else {}
    report=report or {}
    if termination == 'task_complete':
        return 'completed'
    if termination == 'agent_blocked':
        return 'blocked'
    actual=trace.get('actual_outputs') if isinstance(trace.get('actual_outputs'), dict) else {}
    if termination in {'timeout','error_unrecoverable','provider_usage_limit','verifier_failed'} or actual.get('status') in {'failed','timeout','error'}:
        return 'failed'
    if report.get('fallback_trace') or md.get('fallback_trace') or md.get('fallback_trace_created') or md.get('fallback_reason') or report.get('fallback_reason') or (isinstance(report.get('metadata_json'), dict) and report['metadata_json'].get('fallback_reason')):
        return 'fallback'
    steps=trace.get('steps') or []
    if any(s.get('step_outcome') == 'blocked' for s in steps if isinstance(s, dict)):
        return 'blocked'
    if any(s.get('step_outcome') == 'failed' or s.get('success') is False for s in steps if isinstance(s, dict)):
        return 'failed' if not any(s.get('success') is True for s in steps if isinstance(s, dict)) else 'partial'
    if steps:
        return 'partial'
    return 'failed'

def normalize_trace_for_export(trace: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
    out=dict(trace)
    steps=[]
    for s in out.get('steps') or []:
        if not isinstance(s, dict):
            steps.append(s); continue
        ns=dict(s)
        meta=dict(ns.get('metadata_json') or {}) if isinstance(ns.get('metadata_json'), dict) else {}
        command=ns.get('input') or ns.get('command') or meta.get('command')
        op=_operation_from_command(str(command) if command is not None else None, meta)
        if op and (not ns.get('operation') or ns.get('operation') == 'other'):
            ns['operation']=op
        if not ns.get("environment_domain"):
            ns["environment_domain"] = _infer_environment_domain(ns.get("tool"), ns.get("operation"), ns.get("input"), ns.get("output"))
        if ns.get("artifact_refs"):
            ns["artifact_refs"] = _normalize_ref_list(ns.get("artifact_refs"))
        ns['metadata_json']=meta
        steps.append(ns)
    out['steps']=steps
    actual = out.get("actual_outputs") if isinstance(out.get("actual_outputs"), dict) else {}
    attempt_kind = str(out.get("attempt_kind") or actual.get("attempt_kind") or "baseline")
    if actual:
        out.setdefault("actual_outputs_ref", _actual_ref(attempt_kind))
        out.setdefault("output_summary", actual.get("output_summary"))
        out.setdefault("final_output_summary", actual.get("output_summary"))
        out.setdefault("artifact_refs", _normalize_ref_list(actual.get("artifact_refs") or actual.get("files_created")))
        out.setdefault("deliverable_refs", _normalize_ref_list(actual.get("deliverable_refs")))
        out.setdefault("input_artifact_refs", _normalize_ref_list(actual.get("input_artifact_refs")))
    out['attempt_status']=infer_attempt_status(out, report)
    return out

def _tool(command: str | None, raw: dict[str,Any], op: str) -> str | None:
    c=(command or '').strip()
    low=c.lower()
    if low.startswith('python ') or low.startswith('python3 ') or low == 'python' or low == 'python3': return 'python'
    if low.startswith('apply_patch') or low == 'apply_patch': return 'apply_patch'
    if c and any(tok in low for tok in ['sed','cat','ls','rg','find','pwd','mkdir','bash','zsh','sh ','python','touch','cp ','mv ','rm ']): return 'Bash'
    if op in READ_OPS: return 'file_read'
    if op in WRITE_OPS: return 'file_write'
    return raw.get('tool')

def _derive_observation(step_num: int, op: str) -> str:
    if step_num == 1: return 'Attempt workspace and task inputs were available.'
    if op in READ_OPS: return 'Input file was available for inspection.'
    if op in EXEC_OPS or op in WRITE_OPS: return 'Prior inputs and task requirements were available.'
    if op in VERIFY_OPS: return 'Generated outputs were available for validation.'
    return 'Prior workflow context was available.'

def _normalize_outcome(v: Any, meta: dict[str,Any]) -> str | None:
    if v is None: return None
    low=str(v).lower().strip()
    if low in VALID_OUTCOMES: return low
    meta['original_step_outcome']=v
    if low in {'ok','success','succeeded','done'}: return 'completed'
    if low in {'fail','failure','errored','error'}: return 'failed'
    if low in {'block','blocked'}: return 'blocked'
    return 'neutral'


def _safe_int(v: Any, default: int, field: str, meta: dict[str, Any], report: TraceNormalizationReport | None = None) -> int:
    try:
        if isinstance(v, bool):
            raise ValueError('boolean is not numeric')
        if v is None or v == '':
            return default
        return int(v)
    except Exception:
        if v is not None:
            meta[f'original_{field}']=v
            warning=f'{field} value {v!r} was not numeric; defaulted to {default}'
            meta.setdefault('normalization_warnings', []).append(warning)
            if report is not None:
                report.warnings.append(warning)
        return default

def _normalize_step(raw_step: Any, ordinal: int, used_steps: set[int], report: TraceNormalizationReport) -> dict[str,Any]:
    raw = raw_step if isinstance(raw_step, dict) else {'value': raw_step}
    meta=dict(raw.get('metadata_json') or {}) if isinstance(raw.get('metadata_json'), dict) else {}
    meta['raw_step']=raw
    unknown={k:v for k,v in raw.items() if k not in CANONICAL_STEP and k not in {'step_number','index','command','inputs','outputs','state_changes','artifacts','artifact_paths','decision_summary','stdout_summary','exit_code'}}
    if unknown: meta['original_fields']=unknown
    orig_step=raw.get('step', raw.get('step_number', raw.get('index')))
    step_num=_safe_int(orig_step, ordinal, 'step', meta, report)
    if step_num < 1 or step_num in used_steps:
        meta['original_step_number']=orig_step
        step_num=ordinal
        report.warnings.append(f'step {ordinal} renumbered from {orig_step!r}')
    used_steps.add(step_num)
    raw_action=raw.get('action')
    command=raw.get('command')
    inputs=raw.get('inputs')
    raw_input_probe=raw.get('input')
    commands_list = None
    if isinstance(raw_input_probe, dict):
        commands_list = raw_input_probe.get('commands')
    if commands_list is None and isinstance(inputs, dict):
        commands_list = inputs.get('commands')
    inferred_command = command or (raw_input_probe.get('cmd') if isinstance(raw_input_probe, dict) else None) or (raw_input_probe if isinstance(raw_input_probe, str) else None) or (inputs.get('cmd') if isinstance(inputs, dict) else None) or (inputs if isinstance(inputs, str) else None)
    if inferred_command is None and isinstance(commands_list, list):
        inferred_command = '\n'.join(str(c) for c in commands_list)
    op=_operation(str(raw_action) if raw_action is not None else None, raw.get('operation'), meta, report)
    ctx_op=_operation_from_step_context(raw_action, raw.get('operation'), raw.get('tool'), inferred_command, meta, report)
    cmd_op=_operation_from_command(str(inferred_command) if inferred_command is not None else None, meta, report) or ctx_op
    if cmd_op and (raw.get('operation') is None or op == 'other'):
        if op != cmd_op and report is not None:
            report.operation_repair_count += 1
        op=cmd_op
    action=_action(str(raw_action) if raw_action is not None else None, op, meta, report)
    if action == 'error' and raw.get('success') is True and not raw.get('error_type') and not raw.get('error_message'):
        action='agent_step'

    raw_input=raw.get('input')
    input_val: str | None
    if command is not None:
        input_val=str(command)
        meta['command']=command
        if inputs is not None: meta['inputs']=inputs
        if isinstance(raw_input, dict): meta.setdefault('input', raw_input); report.input_dict_to_string_count += 1
    elif isinstance(raw_input, dict):
        meta['inputs']=raw_input
        report.input_dict_to_string_count += 1
        input_val=str(raw_input.get('cmd')) if raw_input.get('cmd') is not None else _compact(raw_input)
    elif raw_input is not None:
        input_val=_string_or_none(raw_input, meta, 'input')
    elif inputs is not None:
        meta['inputs']=inputs
        if isinstance(inputs, dict) and inputs.get('cmd') is not None:
            input_val=str(inputs.get('cmd'))
        else:
            input_val=_compact(inputs)
        if isinstance(inputs, dict): report.input_dict_to_string_count += 1
    else:
        input_val=None

    raw_output=raw.get('output')
    outputs=raw.get('outputs')
    output_val: str | None = None
    if isinstance(raw_output, dict):
        meta['outputs']=raw_output
        output_val=_output_to_string(raw_output)
        report.output_dict_to_string_count += 1
    elif raw_output is not None:
        output_val=_string_or_none(raw_output, meta, 'output')
    if outputs is not None:
        meta['outputs']=outputs
        if isinstance(outputs, dict):
            report.output_dict_to_string_count += 1
        output_val=output_val or _output_to_string(outputs)
    if raw.get('stdout_summary') is not None:
        meta['stdout_summary']=raw.get('stdout_summary')
        output_val=output_val or str(raw.get('stdout_summary'))

    state_change=_string_or_none(raw.get('state_change'), meta, 'state_change')
    if raw.get('state_changes') is not None:
        sc=raw.get('state_changes'); meta['state_changes']=sc
        if isinstance(sc, list):
            state_change=state_change or ('; '.join(str(x) for x in sc) if sc else None)
        else:
            state_change=state_change or _compact(sc)
    artifact_refs=list(raw.get('artifact_refs') or [])
    if raw.get('artifacts') is not None:
        meta['original_artifacts']=raw.get('artifacts')
        artifact_refs.extend(str(x) for x in _as_list(raw.get('artifacts')))
    if raw.get('artifact_paths') is not None:
        meta['original_artifact_paths']=raw.get('artifact_paths')
        artifact_refs.extend(str(x) for x in _as_list(raw.get('artifact_paths')))
    artifact_refs = _normalize_ref_list(artifact_refs)
    if raw.get('exit_code') is not None: meta['exit_code']=raw.get('exit_code')
    if raw.get('decision_summary') is not None: meta['decision_summary']=raw.get('decision_summary')

    observation=_string_or_none(raw.get('observation'), meta, 'observation')
    if observation is None and action != 'user_message':
        observation=_derive_observation(step_num, op); meta['observation_derived']=True
    reasoning=None if action=='user_message' else _string_or_none(raw.get('reasoning') or raw.get('decision_summary'), meta, 'reasoning')

    caused=raw.get('caused_by')
    warnings=[]
    if caused is None:
        caused_by=None if ordinal == 1 else [ordinal-1]
        causal_type=None if ordinal == 1 else 'execution_of_plan'
        causal_note=None if ordinal == 1 else 'Sequential workflow step following the prior action.'
    else:
        caused_by=[int(x) for x in _as_list(caused) if isinstance(x, int) or str(x).isdigit()]
        valid=[x for x in caused_by if 1 <= x < step_num]
        if len(valid) != len(caused_by): warnings.append('invalid caused_by references removed')
        caused_by=valid or None
        causal_type=_normalize_causal_type(raw.get('causal_type') or ('execution_of_plan' if caused_by else None), meta, report)
        causal_note=_string_or_none(raw.get('causal_note'), meta, 'causal_note')
    if caused is None and causal_type is not None:
        causal_type=_normalize_causal_type(causal_type, meta, report)
    if warnings:
        meta['normalization_warnings']=warnings; report.warnings.extend(warnings)

    actor=_actor_to_string(raw.get('actor'), action, meta)
    raw_execution_mode = raw.get('execution_mode') if raw.get('execution_mode') in {'serial','parallel'} else None
    execution_mode = None if action == 'user_message' else (raw_execution_mode or 'serial')
    parallel_group = _string_or_none(raw.get('parallel_group'), meta, 'parallel_group') if execution_mode == 'parallel' else None
    tool = None if action=='user_message' else _tool(command if command is not None else input_val, raw, op)
    environment_domain = None if action == 'user_message' else (
        raw.get('environment_domain') if raw.get('environment_domain') in ENVIRONMENT_DOMAINS else _infer_environment_domain(tool, op, input_val, output_val)
    )
    interaction_surface = None if action == "user_message" else _infer_interaction_surface(raw, tool, op, input_val, output_val)
    feedback_type = raw.get('feedback_type') if raw.get('feedback_type') in {'correction','approval','clarification','new_instruction','other'} else None
    feedback_content = _string_or_none(raw.get('feedback_content'), meta, 'feedback_content')
    step={
        'step':step_num,'turn':_safe_int(raw.get('turn'), 1, 'turn', meta, report),'actor':actor,'action':action,
        'operation': None if action=='user_message' else op,'tool': tool,
        'environment_domain': environment_domain,
        'execution_mode':execution_mode,'parallel_group':parallel_group,'observation':None if action=='user_message' else observation,
        'input':input_val,'input_source':_input_source(raw.get('input_source'), meta),'output':None if action=='user_message' else output_val,
        'state_change':state_change,'reasoning':None if action=='user_message' else reasoning,
        'caused_by':caused_by,'causal_type':causal_type,'causal_note':causal_note,'alternatives_considered':_string_or_none(raw.get('alternatives_considered'), meta, 'alternatives_considered'),
        'success':None if action=='user_message' else raw.get('success') if isinstance(raw.get('success'), bool) else None,'step_outcome':None if action=='user_message' else _normalize_outcome(raw.get('step_outcome'), meta),
        'error_type':_string_or_none(raw.get('error_type'), meta, 'error_type'),'error_message':_string_or_none(raw.get('error_message'), meta, 'error_message'),'message_role':raw.get('message_role') if action=='user_message' else None,
        'feedback_type':feedback_type,'feedback_content':feedback_content,
        'started_at':_string_or_none(raw.get('started_at'), meta, 'started_at'),'ended_at':_string_or_none(raw.get('ended_at'), meta, 'ended_at'),'retry_of':raw.get('retry_of') if isinstance(raw.get('retry_of'), int) else None,'artifact_refs':artifact_refs,'metadata_json':meta,
    }
    if action != "user_message":
        step["interaction_surface"] = interaction_surface
        if interaction_surface and interaction_surface != "unknown" and environment_domain == "unknown":
            if interaction_surface in {"cli"}:
                step["environment_domain"] = "terminal"
            elif interaction_surface in {"mcp_tool", "mcp_resource", "mcp_prompt"}:
                step["environment_domain"] = "mcp"
            elif interaction_surface in {"cloud_console", "saas_app"}:
                step["environment_domain"] = "software"
            elif interaction_surface in {"document", "pdf", "table"}:
                step["environment_domain"] = "multimodal" if interaction_surface in {"document", "pdf"} else "spreadsheet"
            elif interaction_surface in {"device", "industrial", "physical"}:
                step["environment_domain"] = "industrial" if interaction_surface == "industrial" else interaction_surface
            elif interaction_surface in ENVIRONMENT_DOMAINS:
                step["environment_domain"] = interaction_surface
        for field in SURFACE_STEP_FIELDS:
            if field in {"interaction_surface", "surface_capture_status", "missing_surface_fields", "surface_capture_notes"}:
                continue
            if field in raw:
                value = _normalize_surface_value(field, raw.get(field), meta)
                if value not in (None, "", [], {}):
                    step[field] = value
        if command is not None and not step.get("command"):
            step["command"] = str(command)
        elif interaction_surface == "cli" and input_val and not step.get("command"):
            step["command"] = input_val
        if interaction_surface == "file":
            if not step.get("file_path") and artifact_refs:
                step["file_path"] = artifact_refs[0]
            if not step.get("file_operation") and op:
                step["file_operation"] = op
        if raw.get("exit_code") is not None and step.get("exit_code") is None:
            try:
                step["exit_code"] = int(raw.get("exit_code"))
            except (TypeError, ValueError):
                pass
        if raw.get("stdout_ref") is None and raw.get("stdout_summary") is not None:
            meta.setdefault("surface_stdout_summary", raw.get("stdout_summary"))
        if step.get("environment_domain") == "mixed" and not step.get("connected_surfaces") and interaction_surface:
            step["connected_surfaces"] = [interaction_surface]
        status, missing = _surface_completeness(step, meta)
        if step.get("surface_capture_status") is None:
            step["surface_capture_status"] = status
        if status in {"partial", "unavailable"} and missing and not step.get("surface_capture_notes"):
            step["surface_capture_notes"] = f"{interaction_surface} surface evidence is incomplete; missing fields were recorded without fabrication."
        if missing and not step.get("missing_surface_fields"):
            step["missing_surface_fields"] = missing
            meta.setdefault("trace_quality_warnings", []).append(f"{interaction_surface}_missing_surface_fields:{','.join(missing)}")
        if step.get("success") is True and not any(step.get(name) for name in ["output", "state_change", "artifact_refs", "evidence_refs", "raw_response_ref", "response_payload_ref", "screenshot_ref", "visible_state_ref", "telemetry_ref"]):
            meta.setdefault("trace_quality_warnings", []).append("successful_step_missing_output_state_or_evidence")
    if action == 'user_message':
        step.update({'operation':None,'tool':None,'environment_domain':None,'execution_mode':None,'observation':None,'reasoning':None,'success':None,'step_outcome':None,'output':None})
    return step

def _model_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return None
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump(mode='json')
        except TypeError:
            return obj.model_dump()
    return obj

def _valid_actual_outputs(raw_actual: Any, context: TraceNormalizationContext, metadata: dict[str, Any]) -> dict[str, Any] | None:
    if context.actual_outputs is not None:
        if raw_actual is not None:
            metadata['original_actual_outputs']=raw_actual
        return _model_dump(context.actual_outputs)
    if isinstance(raw_actual, dict):
        try:
            return ActualOutputs.model_validate(raw_actual).model_dump()
        except Exception:
            metadata['original_actual_outputs']=raw_actual
            return None
    if raw_actual is not None:
        metadata['original_actual_outputs']=raw_actual
    return None

def _canonical_attempt_kind(raw: dict[str, Any], context: TraceNormalizationContext, metadata: dict[str, Any]) -> str:
    val=raw.get('attempt_kind') or raw.get('attempt') or context.attempt_kind
    if val in VALID_ATTEMPT_KINDS:
        return val
    if val is not None:
        metadata['original_attempt_kind']=val
    return context.attempt_kind if context.attempt_kind in VALID_ATTEMPT_KINDS else 'other'

def _canonical_trace_mode(raw: dict[str, Any], metadata: dict[str, Any]) -> str:
    val=raw.get('trace_mode') or 'live'
    if val in VALID_TRACE_MODES:
        return val
    metadata['original_trace_mode']=val
    return 'live'

def _canonical_termination_reason(raw: dict[str, Any], success_like: bool, metadata: dict[str, Any]) -> str:
    val=raw.get('termination_reason')
    if val in VALID_TERMINATION_REASONS:
        return val
    if val is not None:
        metadata['original_termination_reason']=val
    return 'task_complete' if success_like else 'other'

def _canonical_agent_tools(raw: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    val=raw.get('agent_tools')
    if isinstance(val, list) and all(isinstance(x, str) for x in val):
        return val
    if val is not None:
        metadata['original_agent_tools']=val
    return ['codex_cli','Bash','python','file_read','file_write','apply_patch']

def _canonical_artifacts(raw: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    val=raw.get('artifacts')
    if not val:
        return []
    metadata['original_artifacts']=val
    return []

def _extract_raw_steps(raw: dict[str, Any]) -> tuple[list[Any], str | None]:
    for key in ['steps','trace_steps','trace','records','events','actions']:
        val=raw.get(key)
        if isinstance(val, list):
            return val, key
        if isinstance(val, dict):
            nested=val.get('steps') or val.get('events') or val.get('actions')
            if isinstance(nested, list):
                return nested, key
    return [], None

def normalize_agent_trace(raw: dict[str,Any], context: TraceNormalizationContext) -> dict[str,Any]:
    report=TraceNormalizationReport(task_id=context.task_id, attempt_id=context.attempt_id, attempt_kind=context.attempt_kind)
    raw_steps, raw_step_source = _extract_raw_steps(raw)
    if not isinstance(raw_steps, list): raw_steps=[]; report.warnings.append('raw steps was not a list')
    metadata=dict(raw.get('metadata_json') or {}) if isinstance(raw.get('metadata_json'), dict) else {}
    if raw_step_source:
        metadata['raw_trace_step_source']=raw_step_source
    unknown_top={k:v for k,v in raw.items() if k not in CANONICAL_TOP and k not in {'attempt','role','schema_name'}}
    if unknown_top: metadata['original_fields']=unknown_top
    if raw.get('role') is not None: metadata['original_role']=raw.get('role')
    if raw.get('schema_name') is not None: metadata['schema_name']=raw.get('schema_name')
    used=set(); steps=[_normalize_step(s, i+1, used, report) for i,s in enumerate(raw_steps)]
    # Ensure final monotonic sequence without dropping; preserve original numbers already in metadata when changed.
    if [s['step'] for s in steps] != list(range(1,len(steps)+1)):
        for i,s in enumerate(steps, 1):
            s['metadata_json'].setdefault('original_step_number', s['step']); s['step']=i
        report.warnings.append('steps renumbered to canonical 1..N order')
    actual_outputs = _valid_actual_outputs(raw.get('actual_outputs'), context, metadata)
    success_like=bool(actual_outputs or raw.get('actual_outputs') or any(s.get('success') for s in steps))
    metadata.update({'raw_trace_preserved':True,'discarded_step_count':0,'trace_normalization_status':'normalized' if not report.warnings else 'partial'})
    attempt_kind = _canonical_attempt_kind(raw, context, metadata)
    raw_agent_config = raw.get('agent_config') if isinstance(raw.get('agent_config'), dict) else None
    actual_artifact_refs = _normalize_ref_list((actual_outputs or {}).get('artifact_refs') or (actual_outputs or {}).get('files_created'))
    actual_deliverable_refs = _normalize_ref_list((actual_outputs or {}).get('deliverable_refs'))
    input_artifact_refs = _normalize_ref_list(raw.get('input_artifact_refs') or (actual_outputs or {}).get('input_artifact_refs'))
    environment_domains = {
        s.get("environment_domain")
        for s in steps
        if s.get("environment_domain") and s.get("environment_domain") != "unknown"
    }
    top_environment_domain = raw.get("environment_domain") if raw.get("environment_domain") in ENVIRONMENT_DOMAINS else (
        environment_domains.pop() if len(environment_domains) == 1 else ("mixed" if environment_domains else "unknown")
    )
    missing_runtime_fields = []
    for field in ("run_id", "package_id", "bundle_id", "started_at", "ended_at", "agent_config"):
        if raw.get(field) in (None, "", {}):
            missing_runtime_fields.append(field)
    if missing_runtime_fields:
        metadata.setdefault("missing_runtime_fields", missing_runtime_fields)
    trace={
        'schema_version': str(raw.get('schema_version') or 'aa-trace-v0.1'),
        'trace_id': str(raw.get('trace_id') or f'{context.trace_id_prefix}_{context.attempt_id}_normalized'),
        'collection_id': _string_or_none(raw.get('collection_id'), metadata, 'collection_id'),
        'run_id': _string_or_none(raw.get('run_id'), metadata, 'run_id'),
        'package_id': _string_or_none(raw.get('package_id'), metadata, 'package_id'),
        'bundle_id': _string_or_none(raw.get('bundle_id'), metadata, 'bundle_id'),
        'prior_trace_id': _string_or_none(raw.get('prior_trace_id'), metadata, 'prior_trace_id'), 'trace_mode': _canonical_trace_mode(raw, metadata),
        'task': _string_or_none(raw.get('task'), metadata, 'task') or context.task, 'task_id': _string_or_none(raw.get('task_id'), metadata, 'task_id') or context.task_id, 'task_family_id': _string_or_none(raw.get('task_family_id'), metadata, 'task_family_id'),
        'attempt_id': _string_or_none(raw.get('attempt_id'), metadata, 'attempt_id') or context.attempt_id, 'attempt_kind': attempt_kind,
        'agent_tools': _canonical_agent_tools(raw, metadata),
        'started_at': _string_or_none(raw.get('started_at'), metadata, 'started_at'), 'ended_at': _string_or_none(raw.get('ended_at'), metadata, 'ended_at'), 'system_prompt': _string_or_none(raw.get('system_prompt'), metadata, 'system_prompt'), 'system_prompt_hash': _string_or_none(raw.get('system_prompt_hash'), metadata, 'system_prompt_hash'), 'skills': raw.get('skills') if isinstance(raw.get('skills'), list) else None, 'memory': _string_or_none(raw.get('memory'), metadata, 'memory'), 'agent_config': raw_agent_config,
        'environment_domain': top_environment_domain,
        'actual_outputs_ref': _string_or_none(raw.get('actual_outputs_ref'), metadata, 'actual_outputs_ref') or _actual_ref(attempt_kind),
        'input_artifact_refs': input_artifact_refs,
        'artifact_refs': _normalize_ref_list(raw.get('artifact_refs')) or actual_artifact_refs,
        'deliverable_refs': _normalize_ref_list(raw.get('deliverable_refs')) or actual_deliverable_refs,
        'output_summary': _string_or_none(raw.get('output_summary'), metadata, 'output_summary') or (actual_outputs or {}).get('output_summary'),
        'final_output_summary': _string_or_none(raw.get('final_output_summary'), metadata, 'final_output_summary') or (actual_outputs or {}).get('output_summary'),
        'learning': _string_or_none(raw.get('learning'), metadata, 'learning'),
        'termination_reason': _canonical_termination_reason(raw, success_like, metadata),
        'steps': steps, 'actual_outputs': actual_outputs, 'artifacts': _canonical_artifacts(raw, metadata), 'metadata_json': metadata,
    }
    trace['attempt_status']=infer_attempt_status(trace)
    report.raw_step_count=len(raw_steps); report.normalized_step_count=len(steps); report.discarded_step_count=0
    valid, errors = _validate_normalized_trace(trace)
    report.trace_schema_valid=valid; trace['metadata_json']['schema_valid']=valid
    if not valid:
        report.trace_normalization_partial=True; report.validation_errors.extend(errors)
    report.trace_normalized=True; report.trace_lossless=(report.raw_step_count==report.normalized_step_count and report.discarded_step_count==0)
    return trace

def _validation_error_messages(exc: Exception) -> list[str]:
    if hasattr(exc, 'errors'):
        try:
            return [_compact(e, 2000) for e in exc.errors()]
        except Exception:
            pass
    return [str(exc)]

def _validate_normalized_trace(trace: dict[str, Any]) -> tuple[bool, list[str]]:
    try:
        AgentTrace.model_validate(trace)
        return True, []
    except Exception as exc:
        return False, _validation_error_messages(exc)

def repair_agent_trace_file(raw_trace_path: Path, context: TraceNormalizationContext) -> NormalizationResult:
    report=TraceNormalizationReport(task_id=context.task_id, attempt_id=context.attempt_id, attempt_kind=context.attempt_kind, raw_trace_ref=raw_trace_path.name)
    try:
        text = raw_trace_path.read_text()
        try:
            raw=json.loads(text)
        except json.JSONDecodeError as strict_error:
            raw=json.loads(text, strict=False)
            report.warnings.append(f"raw trace parsed with permissive JSON control-character handling: {strict_error}")
        if isinstance(raw, list):
            raw={'steps': raw, 'metadata_json': {'raw_trace_shape': 'list', 'raw_trace_list': raw}}
        elif isinstance(raw, dict):
            raw_steps, source = _extract_raw_steps(raw)
            if source and source != 'steps':
                md=dict(raw.get('metadata_json') or {}) if isinstance(raw.get('metadata_json'), dict) else {}
                md.setdefault('raw_trace_shape', source)
                raw['metadata_json']=md
                raw['steps']=raw_steps
        else:
            raise ValueError('raw trace JSON was not an object or step list')
    except Exception as e:
        report.raw_trace_parse_error=True; report.validation_errors.append(str(e))
        return NormalizationResult(normalized_trace=None, report=report, fallback_required=True, parse_error=str(e))
    trace=normalize_agent_trace(raw, context)
    valid, errors = _validate_normalized_trace(trace)
    raw_steps, _source = _extract_raw_steps(raw)
    raw_step_count = len(raw_steps) if isinstance(raw_steps, list) else 0
    normalized_step_count = len(trace.get('steps') or [])
    comp=TraceNormalizationReport(task_id=context.task_id, attempt_id=context.attempt_id, attempt_kind=context.attempt_kind, raw_trace_ref=raw_trace_path.name, normalized_trace_ref='agent_trace.normalized.json', canonical_trace_ref='agent_trace.json', trace_schema_valid=valid, trace_normalized=True, trace_lossless=True, fallback_trace=False, raw_step_count=raw_step_count, normalized_step_count=normalized_step_count, discarded_step_count=0, trace_normalization_partial=(not valid) or trace.get('metadata_json',{}).get('trace_normalization_status')=='partial')
    comp.warnings.extend(report.warnings)
    # Re-run normalization into a temporary report to retain repair counters.
    counter_report=TraceNormalizationReport(task_id=context.task_id, attempt_id=context.attempt_id, attempt_kind=context.attempt_kind)
    used=set()
    for i, step in enumerate(raw_steps if isinstance(raw_steps, list) else [], 1):
        _normalize_step(step, i, used, counter_report)
    comp.input_dict_to_string_count=counter_report.input_dict_to_string_count
    comp.output_dict_to_string_count=counter_report.output_dict_to_string_count
    comp.causal_type_repair_count=counter_report.causal_type_repair_count
    comp.operation_repair_count=counter_report.operation_repair_count
    comp.action_repair_count=counter_report.action_repair_count
    comp.warnings.extend(counter_report.warnings)
    comp.trace_lossless = comp.raw_step_count == comp.normalized_step_count and comp.discarded_step_count == 0
    if not comp.trace_lossless: comp.validation_errors.append('normalized step count is less than raw step count')
    if not valid:
        comp.validation_errors.extend(errors)
    trace.setdefault('metadata_json', {})['schema_valid']=valid
    trace['metadata_json']['trace_normalization_status']='normalized' if valid and not comp.warnings else 'partial'
    return NormalizationResult(normalized_trace=trace if valid else trace, report=comp, fallback_required=False)
