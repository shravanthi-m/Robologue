"""Hard runtime limits, including conservative model-call reservations."""
from dataclasses import asdict, dataclass
import math


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeLimits:
    max_episode_steps: int = 600
    max_proposals: int = 10
    max_model_calls: int = 10
    max_model_tokens: int = 100000
    max_cost_usd: float = 1.0
    max_seconds: float = 300.0
    max_input_bytes: int = 16000
    max_output_tokens: int = 1024
    max_call_cost_usd: float = 0.10

    def __post_init__(self):
        for key in ('max_episode_steps', 'max_proposals', 'max_model_calls',
                    'max_model_tokens', 'max_input_bytes', 'max_output_tokens'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if not 1 <= self.max_episode_steps <= 600 or self.max_input_bytes < 1:
            raise ValueError('Invalid episode or input budget')
        for key in ('max_cost_usd', 'max_seconds', 'max_call_cost_usd'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.max_seconds <= 0:
            raise ValueError('max_seconds must be positive')

    def to_dict(self):
        return asdict(self)

    def reserve_model(self, ledger, input_bytes, timeout_seconds):
        """Reserve worst-case cost/tokens BEFORE dispatch, never refund on crash.

        Providers must enforce the supplied output limit and request timeout
        and supply a pricing upper bound. Bytes conservatively bound input
        tokens; the caller is responsible for including all request context.
        """
        if input_bytes > self.max_input_bytes:
            raise BudgetExceeded('Model context exceeds the input limit')
        tokens = input_bytes + self.max_output_tokens
        if ledger['model_calls'] + 1 > self.max_model_calls:
            raise BudgetExceeded('Model call budget exhausted')
        if ledger['reserved_tokens'] + tokens > self.max_model_tokens:
            raise BudgetExceeded('Model token budget exhausted')
        cost = math.ceil(self.max_call_cost_usd * 1000000)
        if ledger['reserved_cost_microusd'] + cost > math.floor(self.max_cost_usd * 1000000):
            raise BudgetExceeded('Model cost budget exhausted')
        ledger['model_calls'] += 1
        ledger['reserved_tokens'] += tokens
        ledger['reserved_cost_microusd'] += cost
        return {'max_output_tokens': self.max_output_tokens,
                'timeout_seconds': timeout_seconds,
                'max_cost_usd': self.max_call_cost_usd,
                'max_input_tokens': input_bytes}


def empty_ledger():
    return {'proposals': 0, 'model_calls': 0, 'reserved_tokens': 0,
            'reserved_cost_microusd': 0}
