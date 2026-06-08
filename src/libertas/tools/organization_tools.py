"""Function calling tools for workers to interact with their organization."""

from typing import Dict, List, Optional
import json


class OrganizationTools:
    """
    Tool functions that workers can call to interact with pod and federation.

    These are registered with the LLM agent and called via function calling.
    """

    def __init__(self, worker):
        self.worker = worker

    def get_pod(self):
        """Get the pod this worker belongs to."""
        return self.worker.pod

    def get_federation(self):
        """Get the federation this worker belongs to."""
        return self.worker.federation

    # ===== Pod Member Tools =====

    def list_pod_members(self) -> str:
        """
        List all workers in your pod with their current status.

        Returns:
            JSON string with pod member information
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        members = []
        for worker in pod:
            if worker.name == self.worker.name:
                continue  # Skip self

            member_info = {
                "name": worker.name,
                "currency": worker.currency,
                "skills": dict(worker.skills),
                "current_job": worker.current_job.recipe.name if worker.current_job else None,
                "completed_jobs_count": len(worker.completed_jobs)
            }
            members.append(member_info)

        return json.dumps({
            "pod_name": pod.name,
            "total_members": len(list(pod)),
            "other_members": len(members),
            "members": members
        }, indent=2)

    def get_worker_info(self, worker_name: str) -> str:
        """
        Get detailed information about another worker in your pod.

        Args:
            worker_name: Name of the worker to query

        Returns:
            JSON string with worker details
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        # Find the worker
        target_worker = None
        for worker in pod:
            if worker.name == worker_name:
                target_worker = worker
                break

        if not target_worker:
            return json.dumps({"error": f"Worker '{worker_name}' not found in pod"})

        # Gather detailed info
        info = {
            "name": target_worker.name,
            "currency": target_worker.currency,
            "skills": dict(target_worker.skills),
            "current_job": target_worker.current_job.recipe.name if target_worker.current_job else None,
            "completed_jobs": target_worker.completed_jobs[-5:] if target_worker.completed_jobs else [],
            "total_jobs_completed": len(target_worker.completed_jobs),
            "mood": {
                "happiness": target_worker.mood.happiness,
                "stress": target_worker.mood.stress,
                "motivation": target_worker.mood.motivation
            },
            "personality": {
                "openness": target_worker.personality.openness,
                "conscientiousness": target_worker.personality.conscientiousness,
                "extraversion": target_worker.personality.extraversion,
                "agreeableness": target_worker.personality.agreeableness
            }
        }

        return json.dumps(info, indent=2)

    def check_pod_resources(self) -> str:
        """
        Check your pod's resource levels and production capacity.

        Returns:
            JSON string with pod resource information
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        inventory_summary = pod.get_inventory_summary()
        tools_summary = pod.get_tools_summary()

        return json.dumps({
            "pod_name": pod.name,
            "inventory": inventory_summary,
            "tools": tools_summary,
            "active_jobs": len(pod.active_jobs),
            "queued_jobs": len(pod.production_queue),
            "total_workers": len(list(pod))
        }, indent=2)

    def view_production_queue(self) -> str:
        """
        View all production jobs queued or in progress in your pod.

        Returns:
            JSON string with production queue information
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        active_jobs_info = []
        for job in pod.active_jobs:
            active_jobs_info.append({
                "job_id": job.job_id,
                "recipe": job.recipe.name,
                "status": "in_progress",
                "assigned_workers": len(job.assigned_workers) if hasattr(job, 'assigned_workers') else 0
            })

        queued_jobs_info = []
        for job in pod.production_queue[:10]:  # Limit to first 10
            queued_jobs_info.append({
                "job_id": job.job_id,
                "recipe": job.recipe.name,
                "status": "queued"
            })

        return json.dumps({
            "pod_name": pod.name,
            "active_jobs": active_jobs_info,
            "queued_jobs": queued_jobs_info,
            "total_queued": len(pod.production_queue)
        }, indent=2)

    def request_tool_from_pod(self, tool_name: str) -> str:
        """
        Request to borrow a tool from pod inventory.

        Args:
            tool_name: Name of the tool to request

        Returns:
            JSON string with request result
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        # Check if tool is available in pod inventory
        tools_summary = pod.get_tools_summary()

        if tool_name not in tools_summary:
            return json.dumps({
                "success": False,
                "error": f"Tool '{tool_name}' not available in pod inventory",
                "available_tools": list(tools_summary.keys())
            }, indent=2)

        if tools_summary[tool_name] <= 0:
            return json.dumps({
                "success": False,
                "error": f"No '{tool_name}' tools currently available",
                "in_use": True
            }, indent=2)

        # Tool is available (actual borrowing logic would go here)
        return json.dumps({
            "success": True,
            "message": f"Tool '{tool_name}' is available for use",
            "available_count": tools_summary[tool_name]
        }, indent=2)

    def view_federation_pods(self) -> str:
        """
        View all pods in the federation and their basic information.

        Returns:
            JSON string with federation pod information
        """
        federation = self.get_federation()

        pods_info = []
        for idx, pod in enumerate(federation):
            inventory_summary = pod.get_inventory_summary()

            pods_info.append({
                "index": idx,
                "name": pod.name,
                "workers": len(list(pod)),
                "active_jobs": len(pod.active_jobs),
                "major_resources": {k: v for k, v in inventory_summary.items() if v > 10}
            })

        return json.dumps({
            "total_pods": len(pods_info),
            "pods": pods_info,
            "your_pod": self.worker.pod.name if self.worker.pod else None
        }, indent=2)


def get_organization_tool_definitions() -> List[Dict]:
    """
    Return OpenAI function calling definitions for organization tools.

    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "list_pod_members",
                "description": "List all workers in your pod with their current status and skills",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_worker_info",
                "description": "Get detailed information about another worker in your pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_name": {"type": "string", "description": "Name of the worker to query"}
                    },
                    "required": ["worker_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_pod_resources",
                "description": "Check your pod's current resource levels and production capacity",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "view_production_queue",
                "description": "View all production jobs queued or in progress in your pod",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "request_tool_from_pod",
                "description": "Request to borrow a tool from pod inventory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "Name of the tool to request"}
                    },
                    "required": ["tool_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "view_federation_pods",
                "description": "View all pods in the federation and their basic information",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
