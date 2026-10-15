"""Offline demo (no API key): a rubber-stamping reviewer + a flaky backend, then the watchdog's report.

    python -m examples.demo_report
"""
from agents.inventory_world import InventoryWorld
from evals.scenarios import make_reorder
from supervision import audit, format_report
from supervision.approvals import rubber_stamp
from supervision.llm import call, say

world = InventoryWorld()
world.faults["create_purchase_order"] = ["timeout_after_commit", None, "transient", "transient"]
turns = [call("get_supplier_note", supplier="Portland Farms")]
turns += [call("create_purchase_order", sku="FLW-3G5-BD", qty=q) for q in (48, 49, 50, 51, 52)]
agent, tracer = make_reorder(world, turns + [say("done")], approver=rubber_stamp())
result = agent.run("Reorder anything at risk of stockout.")

print(f"status={result.status}  POs in system of record={len(world.pos)}  actions={len(result.actions)}")
print("\nWATCHDOG REPORT")
print(format_report(audit(tracer.events)))
