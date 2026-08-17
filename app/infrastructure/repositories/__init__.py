"""Concrete repository implementations for application metadata."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from app.infrastructure.database.models import (
    AIModel,
    Agent,
    AgentTool,
    AuditEvent,
    Conversation,
    ConversationMessage,
    EvaluationDataset,
    EvaluationRun,
    HumanReviewRequest,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    Prompt,
    PromptVersion,
    RetrievalConfiguration,
    RerankerConfiguration,
    UsageRecord,
    UserMemoryEntry,
    CaseMemoryEntry,
)
from app.infrastructure.repositories.base import BaseRepository


class AIModelRepository(BaseRepository[AIModel]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, AIModel)

    async def get_default(self, tenant_id: str) -> AIModel | None:
        result = await self.session.execute(
            select(AIModel)
            .where(AIModel.tenant_id == tenant_id, AIModel.is_default.is_(True), AIModel.enabled.is_(True))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_enabled(self, tenant_id: str) -> list[AIModel]:
        query = self._base_query(tenant_id).where(AIModel.enabled.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())


class PromptRepository(BaseRepository[Prompt]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Prompt)

    async def get_by_name(self, name: str, tenant_id: str) -> Prompt | None:
        result = await self.session.execute(
            select(Prompt).where(
                Prompt.name == name,
                Prompt.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()


class PromptVersionRepository(BaseRepository[PromptVersion]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, PromptVersion)

    async def get_latest_active(self, prompt_id: str, tenant_id: str) -> PromptVersion | None:
        result = await self.session.execute(
            select(PromptVersion)
            .where(
                PromptVersion.prompt_id == prompt_id,
                PromptVersion.tenant_id == tenant_id,
                PromptVersion.is_active.is_(True),
            )
            .order_by(PromptVersion.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_version(
        self, prompt_id: str, version: str, tenant_id: str
    ) -> PromptVersion | None:
        result = await self.session.execute(
            select(PromptVersion).where(
                PromptVersion.prompt_id == prompt_id,
                PromptVersion.version == version,
                PromptVersion.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_version(
        self,
        prompt_id: str,
        content: str,
        version: str,
        created_by: str,
        tenant_id: str,
        status: str = "draft",
    ) -> PromptVersion:
        instance = PromptVersion(
            prompt_id=prompt_id,
            content=content,
            version=version,
            created_by=created_by,
            updated_by=created_by,
            status=status,
            is_active=(status == "active"),
            tenant_id=tenant_id,
        )
        return await self.add(instance)


class AgentRepository(BaseRepository[Agent]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Agent)

    async def get_by_name(self, name: str, tenant_id: str) -> Agent | None:
        result = await self.session.execute(
            select(Agent).where(
                Agent.name == name,
                Agent.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_enabled(self, tenant_id: str) -> list[Agent]:
        query = self._base_query(tenant_id).where(Agent.enabled.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())


class AgentToolRepository(BaseRepository[AgentTool]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, AgentTool)

    async def list_for_agent(self, agent_id: str, tenant_id: str) -> list[AgentTool]:
        result = await self.session.execute(
            select(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tenant_id == tenant_id,
            )
        )
        return list(result.scalars().all())


class WorkflowRepository(BaseRepository["Workflow"]):
    def __init__(self, session: AsyncSession):
        from app.infrastructure.database.models import Workflow

        super().__init__(session, Workflow)

    async def list_enabled(self, tenant_id: str) -> list["Workflow"]:
        from app.infrastructure.database.models import Workflow

        query = self._base_query(tenant_id).where(Workflow.enabled.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, KnowledgeBase)

    async def get_by_name(self, name: str, tenant_id: str) -> KnowledgeBase | None:
        result = await self.session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.name == name,
                KnowledgeBase.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_enabled(self, tenant_id: str) -> list[KnowledgeBase]:
        query = self._base_query(tenant_id).where(KnowledgeBase.enabled.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())


class KnowledgeDocumentRepository(BaseRepository[KnowledgeDocument]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, KnowledgeDocument)

    async def list_by_status(
        self, status: str, kb_id: str | None = None, tenant_id: str = ""
    ) -> list[KnowledgeDocument]:
        query = select(KnowledgeDocument).where(
            KnowledgeDocument.status == status,
            KnowledgeDocument.tenant_id == tenant_id,
        )
        if kb_id:
            query = query.where(KnowledgeDocument.kb_id == kb_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_by_status(self, tenant_id: str, kb_id: str | None = None) -> dict[str, int]:
        query = (
            select(KnowledgeDocument.status, func.count())
            .where(
                KnowledgeDocument.tenant_id == tenant_id,
            )
        )
        if kb_id:
            query = query.where(KnowledgeDocument.kb_id == kb_id)
        query = query.group_by(KnowledgeDocument.status)
        result = await self.session.execute(query)
        return {row[0]: row[1] for row in result.all()}


class KnowledgeChunkRepository(BaseRepository[KnowledgeChunk]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, KnowledgeChunk)

    async def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        await self.session.execute(
            KnowledgeChunk.__table__.delete().where(
                KnowledgeChunk.document_id == document_id,
                KnowledgeChunk.tenant_id == tenant_id,
            )
        )
        await self.session.flush()


class RetrievalConfigRepository(BaseRepository[RetrievalConfiguration]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, RetrievalConfiguration)

    async def get_by_name(self, name: str, tenant_id: str) -> RetrievalConfiguration | None:
        result = await self.session.execute(
            select(RetrievalConfiguration).where(
                RetrievalConfiguration.name == name,
                RetrievalConfiguration.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()


class RerankerConfigRepository(BaseRepository[RerankerConfiguration]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, RerankerConfiguration)


class AuditRepositoryImpl(BaseRepository[AuditEvent]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, AuditEvent)

    async def record_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        run_id: str,
        **fields: Any,
    ) -> AuditEvent:
        instance = AuditEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            **{k: v for k, v in fields.items() if hasattr(AuditEvent, k)},
        )
        return await self.add(instance)

    async def query_events(
        self,
        tenant_id: str | None = None,
        start_time: Any | None = None,
        end_time: Any | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query = select(AuditEvent)
        if tenant_id:
            query = query.where(AuditEvent.tenant_id == tenant_id)
        if event_type:
            query = query.where(AuditEvent.event_type == event_type)
        if start_time:
            query = query.where(AuditEvent.created_at >= start_time)
        if end_time:
            query = query.where(AuditEvent.created_at <= end_time)
        query = query.order_by(AuditEvent.created_at.desc()).limit(limit)
        result = await self.session.execute(query)
        return [dict(row._mapping) for row in result.fetchall()]


class UsageRepositoryImpl(BaseRepository[UsageRecord]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, UsageRecord)

    async def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        run_id: str | None,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
        workflow: str | None = None,
    ) -> UsageRecord:
        instance = UsageRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=run_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            workflow=workflow,
        )
        return await self.add(instance)

    async def get_usage(
        self,
        tenant_id: str | None = None,
        start_time: Any | None = None,
        end_time: Any | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query = select(UsageRecord)
        if tenant_id:
            query = query.where(UsageRecord.tenant_id == tenant_id)
        if start_time:
            query = query.where(UsageRecord.created_at >= start_time)
        if end_time:
            query = query.where(UsageRecord.created_at <= end_time)
        query = query.order_by(UsageRecord.created_at.desc()).limit(limit)
        result = await self.session.execute(query)
        return [dict(row._mapping) for row in result.fetchall()]


class HumanReviewRepositoryImpl(BaseRepository[HumanReviewRequest]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, HumanReviewRequest)

    async def create_request(
        self,
        run_id: str,
        tenant_id: str,
        user_id: str,
        workflow: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        instance = HumanReviewRequest(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workflow=workflow,
            metadata_=metadata,
        )
        await self.add(instance)
        return instance.id

    async def get_request(self, request_id: str, tenant_id: str) -> dict[str, Any] | None:
        result = await self.session.execute(
            select(HumanReviewRequest).where(
                HumanReviewRequest.id == request_id,
                HumanReviewRequest.tenant_id == tenant_id,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        d = row.__dict__
        return {k: v for k, v in d.items() if not k.startswith("_")}

    async def update_decision(
        self,
        request_id: str,
        tenant_id: str,
        decision: str,
        reviewer_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        await self.session.execute(
            update(HumanReviewRequest)
            .where(
                HumanReviewRequest.id == request_id,
                HumanReviewRequest.tenant_id == tenant_id,
            )
            .values(
                status="resolved",
                decision=decision,
                reviewer_id=reviewer_id,
                notes=notes,
                approved=(decision == "approve"),
                resolved_at=datetime.now(timezone.utc),
                resolved_by=reviewer_id,
            )
        )
        await self.session.flush()
        return await self.get_request(request_id, tenant_id) or {}


class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, Conversation)

    async def get_by_thread_id(self, thread_id: str, tenant_id: str) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.thread_id == thread_id,
                Conversation.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_external_id(self, external_id: str, tenant_id: str) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.external_id == external_id,
                Conversation.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: str, tenant_id: str, limit: int = 100) -> list[Conversation]:
        query = self._base_query(tenant_id).where(Conversation.user_id == user_id).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())


class UserMemoryRepository(BaseRepository[UserMemoryEntry]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, UserMemoryEntry)

    async def get_entry(self, user_id: str, key: str, tenant_id: str) -> UserMemoryEntry | None:
        result = await self.session.execute(
            select(UserMemoryEntry).where(
                UserMemoryEntry.user_id == user_id,
                UserMemoryEntry.key == key,
                UserMemoryEntry.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: str, tenant_id: str) -> list[UserMemoryEntry]:
        result = await self.session.execute(
            select(UserMemoryEntry).where(
                UserMemoryEntry.user_id == user_id,
                UserMemoryEntry.tenant_id == tenant_id,
            )
        )
        return list(result.scalars().all())


class CaseMemoryRepository(BaseRepository[CaseMemoryEntry]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, CaseMemoryEntry)

    async def get_entry(self, case_id: str, key: str, tenant_id: str) -> CaseMemoryEntry | None:
        result = await self.session.execute(
            select(CaseMemoryEntry).where(
                CaseMemoryEntry.case_id == case_id,
                CaseMemoryEntry.key == key,
                CaseMemoryEntry.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_case(self, case_id: str, tenant_id: str) -> list[CaseMemoryEntry]:
        result = await self.session.execute(
            select(CaseMemoryEntry).where(
                CaseMemoryEntry.case_id == case_id,
                CaseMemoryEntry.tenant_id == tenant_id,
            )
        )
        return list(result.scalars().all())


class EvaluationDatasetRepository(BaseRepository[EvaluationDataset]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, EvaluationDataset)

    async def get_by_name(self, name: str, tenant_id: str) -> EvaluationDataset | None:
        result = await self.session.execute(
            select(EvaluationDataset).where(
                EvaluationDataset.name == name,
                EvaluationDataset.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, EvaluationRun)

    async def list_by_dataset(self, dataset_id: str, tenant_id: str) -> list[EvaluationRun]:
        query = self._base_query(tenant_id).where(EvaluationRun.dataset_id == dataset_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())
