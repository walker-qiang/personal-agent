#!/usr/bin/env python3
"""通过真实 DeepResearchWorkflow 验收复查协议；使用隔离工具与模型响应。"""
from __future__ import annotations
import copy
import hashlib
import json
from matrix.runtime.adapters.deep_research import DeepResearchWorkflow, _model_evidence
from matrix.runtime.adapters.review import evidence_text
from matrix.runtime.testing.memory_store import MemoryOperationStore
from matrix.chat._service import (_normalize_research_result, _research_result_issues,
    _sanitize_unverified_research_result, _select_latest_official_url, _extract_official_report_period)

DATE='2026-09-14'
URLS=['https://www.sse.com.cn/one.pdf','https://www.sse.com.cn/two.pdf']
QUOTE='The official notice states that the transaction has completed with cash consideration.'
HASH=hashlib.sha256(QUOTE.encode()).hexdigest()

def run(code,kind,*,quote_mode=None,blocked=False):
    snapshot={'version':1,'revision':'fixture-revision','code':code,'object_type':kind,
        'baseline':{'product':{'quote_mode':quote_mode}} if quote_mode else {},
        'gaps':[{'id':str(i),'title':'核验事项'+str(i),'url':u,'kind':'evidence'} for i,u in enumerate(URLS)]}
    class Tools:
        def __init__(self):self.calls=[]
        def tool_names(self):return {'personal_os.'+x for x in ['research_context','market_quote','profile','financials','dividend','valuation','peers','announcements','information_search','web_fetch']}
        def call(self,name,args,**kwargs):
            self.calls.append((name,args))
            if name=='personal_os.research_context':return {'review_context':snapshot,'records':[]}
            if name=='personal_os.market_quote':return {'price':10,'datetime':DATE+'T10:00:00+08:00'}
            if name=='personal_os.financials':return {'data':{'reports':[{'period_end':'2026-06-30','period_type':'H1','currency':'CNY','unit':'yuan','source':{'verification_status':'reconciled'},'statements':[]}],
                'metadata':{'latest_report_period':'2026-06-30','latest_period_type':'H1','stale':False,'source_verification':'reconciled','provenance_ready':True}}}
            if name in {'personal_os.announcements','personal_os.information_search'}:return {'items':[{'title':'2026半年度报告','url':u,'tier':'official','report_period':'2026-06-30'} for u in URLS]}
            if name=='personal_os.web_fetch':
                if blocked:return {'error':'HTTP 403'}
                return {'url':args['url'],'content':QUOTE,'content_hash':HASH,'source_tier':'official','verification_status':'verified','report_period':'2026-06-30'}
            return {'data':'fixture data','estimated':False}
    class LLM:
        def complete_json(self,system,messages,**kwargs):
            return {'schema_version':2,'type':'investment-research','status':'complete','object_type':kind,
                'subject':{'code':code,'name':'验收标的'},'research_date':DATE,'data_date':DATE,'latest_report_period':'2026-06-30',
                'information_completeness':'standard','decision':{'action':'research before action'},
                'summary':'根据官方材料更新经营与产品判断，关注剩余风险。','highlights':['结论一','结论二','结论三'],
                'thesis':['逻辑一','逻辑二','逻辑三'],'antithesis':['反证一','反证二'],'risks':['风险一','风险二','风险三'],
                'metrics':[{'name':'指标'+str(i),'value':str(i),'period':'2026-06-30','source':'fixture'} for i in range(4)],
                'triggers':[{'type':'event','condition':'重大事项变化'},{'type':'financial','condition':'经营变化'}],
                'sources':[{'title':'官方资料','url':u,'date':DATE,'source_type':'official'} for u in URLS],'tags':['fixture'],
                'review_result':{'checks':[{'id':str(i),'title':'核验事项'+str(i),'kind':'evidence','url':u,'status':'resolved',
                    'finding':'公告明确确认交易已经完成并以现金支付，保留后续回报跟踪。','citations':[{'url':u,'content_hash':HASH,'quote':QUOTE}]} for i,u in enumerate(URLS)]}}
    tools=Tools();workflow=DeepResearchWorkflow(MemoryOperationStore(),LLM(),tools,
        normalize_result=_normalize_research_result,validate_result=_research_result_issues,
        sanitize_result=_sanitize_unverified_research_result,preview_json=lambda x:json.dumps(x),
        select_latest_official_url=_select_latest_official_url,extract_official_report_period=_extract_official_report_period)
    question=f'研究对象：验收标的\n标的代码：{code}\n对象类型：{kind}\n研究日期：{DATE}\n复查协议版本：1\n<review_context>'+json.dumps(snapshot,ensure_ascii=False)+'</review_context>'
    handle=workflow.start(owner_id='fixture',session_id='review-'+code,question=question,name='验收标的',research_date=DATE)
    events=list(handle.events());result=handle.result()
    assert result.outcome.value=='completed',(code,result.error)
    parsed=json.loads(result.final_message)
    docs=[args['url'] for name,args in tools.calls if name=='personal_os.web_fetch']
    assert docs==URLS,(code,docs)
    assert tools.calls[0][0]=='personal_os.research_context'
    assert tools.calls[0][1]['object_type']==kind
    checks=parsed['review_result']['checks'];assert parsed['review_result']['context_revision']=='fixture-revision'
    if blocked:
        assert all(x['status']=='pending' for x in checks),checks
        if kind=='fund':assert parsed['status']=='incomplete' and parsed['review_result']['issues'],parsed
    else:
        assert not parsed['review_result']['issues'],parsed['review_result']['issues']
        assert all(x['status']=='resolved' for x in checks),checks
    names={name for name,args in tools.calls}
    if kind=='fund':
        assert not names.intersection({'personal_os.financials','personal_os.peers','personal_os.dividend','personal_os.profile'})
        assert ('personal_os.market_quote' in names)==(quote_mode=='market'),names
    return {'code':code,'kind':kind,'blocked':blocked,'tool_count':len(tools.calls),'result':'passed'}

if __name__=='__main__':
    results=[run('sh600001','stock'),run('000001','fund',quote_mode='nav'),run('VOO','fund',quote_mode='market'),run('fund-blocked','fund',quote_mode='nav',blocked=True)]
    # Verify the model-visible view cannot revive quarantined financial values or lose a tail document.
    raw=[{'tool':'personal_os.financials','result':{'data':{'reports':[{'period_end':'2020-12-31','source':{'verification_status':'unverified'},'statements':[{'secret_unverified':987654321}]}]}}},
         {'tool':'personal_os.web_fetch','result':{'content':'A'*90000,'url':URLS[0]}},
         {'tool':'personal_os.web_fetch','result':{'content':'TAIL_EVIDENCE_MUST_REMAIN','url':URLS[1]}}]
    visible=evidence_text(_model_evidence(raw));json.loads(visible)
    assert '987654321' not in visible and 'TAIL_EVIDENCE_MUST_REMAIN' in visible
    print(json.dumps({'status':'passed','scenarios':results,'evidence_budget_and_quarantine':'passed'},ensure_ascii=False,indent=2))
