"""Small learned action values; environment feedback is supplied explicitly.

Only neural weights/counters persist. No simulator or optimal-action table is
used at inference. Relative displacement and quadratic features are priors.
"""
import torch
from torch import nn


class FeedbackPolicy(nn.Module):
    def __init__(self, radius=1.2):
        super().__init__()
        if not .5 <= radius <= 2:
            raise ValueError('Invalid policy input radius')
        self.radius = radius
        self.head = nn.Linear(5, 5)
        self.register_buffer('feedback_steps', torch.tensor(0, dtype=torch.long))

    def config_dict(self):
        return {'radius': self.radius}

    def forward(self, displacement):
        if displacement.ndim != 2 or displacement.shape[1] != 2 or not bool(torch.isfinite(displacement).all()):
            raise ValueError('Finite relative 2D observation required')
        if bool((displacement.abs() > self.radius).any()):
            raise ValueError('Observation outside the audited policy domain')
        x, y = displacement.unbind(-1)
        return self.head(torch.stack([x, y, x*x, y*y, x*y], -1))

    @torch.no_grad()
    def act(self, position, goal):
        if position.shape != goal.shape:
            raise ValueError('Position and goal shapes differ')
        if int(self.feedback_steps) == 0:
            raise ValueError('Policy has no learned feedback yet')
        return self(goal-position).argmax(-1)


def learn_feedback(policy, optimizer, position, goal, action, outcome, feedback):
    """One finite update from an actual environment tuple, never action labels.

    The environment owns reward semantics. Outcome is validated, but the learner
    does not reconstruct a reward formula or treat movements as symbolic facts.
    """
    if position.shape != goal.shape or outcome.shape != position.shape:
        raise ValueError('Matching observation/outcome batches required')
    if not bool(torch.isfinite(outcome).all()) or action.shape != position.shape[:1] or feedback.shape != action.shape:
        raise ValueError('Invalid feedback tuple')
    if action.dtype != torch.long or bool(((action < 0) | (action >= 5)).any()) or not bool(torch.isfinite(feedback).all()):
        raise ValueError('Invalid action or feedback')
    predictions = policy((goal-position).detach()).gather(1, action[:, None]).squeeze(1)
    loss = (predictions-feedback.detach()).square().mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite feedback loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    policy.feedback_steps.add_(1)
    return float(loss.detach())
