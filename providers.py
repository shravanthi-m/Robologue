"""Person 3 integration boundary: bounded data proposals, never generated code.

    provider.name / provider.is_model
    provider.propose(context, permit) -> ProposalResult

Models receive an allowlisted context. Their adapters must honor permit's
request timeout, output token limit and conservative pricing bound. Runtime
reserves quotas durably before dispatch and gates every returned mutation.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProposalResult:
    mutation: Optional[dict]
    metadata: dict = field(default_factory=dict)


class DeterministicProvider:
    name = 'deterministic'
    is_model = False

    def propose(self, context, permit):
        from evolver.proposer import propose_mutation
        return ProposalResult(propose_mutation(context['pattern'], context['architecture'],
                                               traces=context['traces']))


def proposal_context(pattern, architecture, traces, lessons):
    from skills import SKILLS
    # No credentials, database handles, world truth or evaluation labels.
    return {'pattern': pattern, 'architecture': {
                k: architecture[k] for k in ('version_id', 'active_modules', 'tools',
                                               'policies', 'verification_strategy')},
            'traces': [{'trace_id': t['trace_id'], 'task_id': t['task_id'],
                        'failure_category': t.get('failure_category'),
                        'success': t['success'], 'metrics': t['metrics']}
                       for t in traces],
            'approved_lessons': [{k: lesson[k] for k in (
                'lesson_id', 'trace_id', 'task_id', 'inferred_lesson', 'recommended_component',
                'status') if k in lesson} for lesson in lessons],
            'skills': {k: SKILLS[k] for k in ('navigation', 'collision_check', 'spatial_memory')},
            'allowed_mutations': ['ADD_VERIFIER', 'ADD_MODULE', 'REMOVE_MODULE',
                                  'MODIFY_POLICY', 'MODIFY_RETRY_POLICY']}
