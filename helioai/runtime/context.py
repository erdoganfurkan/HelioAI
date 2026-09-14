"""Who is running, in which session, writing where — as an object, not an ambience.

Three contextvars (`workspace._current_user`, `_current_session`, `_current_label`) used
to be set by each loop, by the MCP server and by nobody in a test, and read back from
inside the tools, the datastore and the sandbox: an argument nobody passed and everybody
depended on. A tool called from a test wrote under the default user; a sub-agent worked
only because the lead had bound the label first; a save landed in a temporary directory
when the caller forgot the session.

`RunContext` carries the same facts explicitly and is handed to the runner, which binds
them for the duration of a run — `bound()` is the one place the three contextvars are
set and reset. The tools that write receive their directories as trusted arguments
(`tool_exec.trusted_args`); the contextvars remain the hot path for everything else,
`RunContext.current()` reads them back, and the fallback logs when it is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import helioai.workspace as _ws


@dataclass(frozen=True)
class RunContext:
    """The facts a run writes under.

    Attributes:
        user_id: Owner of every path derived during the run.
        session_id: The conversation; a sub-agent runs under its parent's.
        session_dir: The workspace directory — figures, scripts, npz, manifest, ledger.
            A sub-agent shares its lead's, which is how `load_data()` reaches what the
            lead downloaded.
        label: The human-readable directory name, when the session has one.
        agent: `"lead"` or the sub-agent role.
        task_id: The delegation's correlation id, for a sub-agent.
        no_network: Whether `run_python` is denied a network namespace.
    """

    user_id: str
    session_id: str
    session_dir: Path
    label: str | None = None
    agent: str = "lead"
    task_id: str | None = None
    no_network: bool = False

    @classmethod
    def for_session(
        cls, user_id: str, session_id: str, *, label: str | None = None, **extra
    ) -> RunContext:
        """Build a context from ids, deriving the session directory the way
        `workspace.get_session_dir` does: by label when there is one, else by id.

        Args:
            user_id: Owner of the session.
            session_id: The conversation.
            label: Its workspace directory name, when already minted.
            **extra: `agent`, `task_id`, `no_network`.
        """
        return cls(
            user_id=user_id,
            session_id=session_id,
            session_dir=_ws.session_dir_for(user_id, session_id, label),
            label=label,
            **extra,
        )

    @classmethod
    def current(cls, **extra) -> RunContext | None:
        """The context the ambient contextvars describe, or None when no session is bound.

        The transition path: a caller that has not been handed a context yet can still
        obtain the one its caller bound.
        """
        session_id = _ws._current_session.get()
        if not session_id:
            return None
        return cls.for_session(
            _ws.current_user(), session_id, label=_ws._current_label.get(), **extra
        )

    def child(self, *, agent: str, task_id: str | None, no_network: bool = False) -> RunContext:
        """A sub-agent's context: the same user, session and directory, its own identity."""
        return replace(self, agent=agent, task_id=task_id, no_network=no_network)

    @property
    def data_dir(self) -> Path:
        """Where the session's downloads (npz + manifest) live."""
        from helioai.datastore import DATA_SUBDIR

        return self.session_dir / DATA_SUBDIR

    @property
    def catalogs_dir(self) -> Path:
        """Where the user's saved catalogues live: beside the workspaces, not in one."""
        return _ws.user_home(self.user_id) / "catalogs"

    @contextmanager
    def bound(self) -> Iterator[RunContext]:
        """Bind the workspace contextvars to this context for the duration of a block.

        Tokens are reset in reverse order, so a sub-agent bound inside its lead's run
        hands the lead's bindings back untouched.
        """
        tokens = [_ws.set_user(self.user_id), _ws.set_session(self.session_id)]
        resets = [_ws.reset_user, _ws.reset_session]
        if self.label:
            tokens.append(_ws.set_label(self.label))
            resets.append(_ws.reset_label)
        try:
            yield self
        finally:
            for reset, token in zip(reversed(resets), reversed(tokens), strict=True):
                reset(token)
