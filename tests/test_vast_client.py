import json
import unittest
from unittest.mock import patch

import requests

from job_utils import Deadline, JobError
from vast_client import (
    ApiError, DEFAULT_IMAGE, DEFAULT_TEMPLATE_HASH, Offer, Template,
    VastClient, choose_offer, merge_filters,
)
from tests.fakes import Clock, Response, Session, VAST_TOKEN, raw_offer


TEMPLATE = Template(DEFAULT_TEMPLATE_HASH, "Fixture", DEFAULT_IMAGE, 12.4, {}, 32)


def fixture_offer(offer_id=1, gpu="A100 SXM4", memory=40960, price=1):
    return Offer(offer_id, gpu, memory, price, 2, 0.01, 0.02)


class SelectionTests(unittest.TestCase):
    def test_price_cap_is_inclusive_and_rejects_actual_over_budget_offers(self):
        allowed = fixture_offer(price=4)
        self.assertEqual(choose_offer([allowed], "prefer-a100", 4), allowed)
        with self.assertRaises(JobError):
            choose_offer([fixture_offer(price=4.0001)], "prefer-a100", 4)
        with self.assertRaises(JobError):
            choose_offer([fixture_offer(price=-1)], "prefer-a100", 4)

    def test_vram_floor_checks_per_gpu_api_units(self):
        allowed = fixture_offer(memory=24000)
        self.assertEqual(choose_offer([allowed], "prefer-a100", 4), allowed)
        with self.assertRaises(JobError):
            choose_offer([fixture_offer(memory=23999)], "prefer-a100", 4)

    def test_a100_preference_and_fallback_are_not_just_cheapest_price(self):
        a100 = fixture_offer(price=3)
        fallback = fixture_offer(2, "RTX 3090", 24576, 0.2)
        self.assertEqual(choose_offer([fallback, a100], "prefer-a100", 4), a100)
        self.assertEqual(choose_offer([fallback], "prefer-a100", 4), fallback)
        with self.assertRaises(JobError):
            choose_offer([fallback], "a100-only", 4)

    def test_template_numeric_constraints_cannot_be_weakened(self):
        merged = merge_filters({"gpu_ram": {"gte": 80000}}, {"gpu_ram": {"gte": 24000}})
        self.assertEqual(merged["gpu_ram"]["gte"], 80000)

    def test_malformed_numeric_template_constraints_are_actionable_errors(self):
        with self.assertRaises(JobError):
            merge_filters({"gpu_ram": {"gte": "invalid"}}, {"gpu_ram": {"gte": 24000}})


class ApiTests(unittest.TestCase):
    def client(self, *responses):
        session = Session(*responses)
        return VastClient(VAST_TOKEN, session=session), session

    def test_pinned_default_template_and_extra_filters_are_parsed(self):
        response = Response({"templates": [{
            "hash_id": DEFAULT_TEMPLATE_HASH, "name": "CUDA", "image": "pytorch/pytorch",
            "tag": "2.4.0-cuda12.4-cudnn9-devel", "recommended_disk_space": 32,
            "extra_filters": json.dumps({"cpu_arch": {"in": ["amd64"]}}),
        }]})
        client, session = self.client(response)
        template = client.template(DEFAULT_TEMPLATE_HASH, Deadline(10))
        self.assertEqual(template.image, DEFAULT_IMAGE)
        self.assertEqual(template.filters["cpu_arch"], {"in": ["amd64"]})
        self.assertEqual(session.headers["Authorization"], "Bearer " + VAST_TOKEN)
        self.assertNotIn(VAST_TOKEN, session.calls[0][1])
        self.assertFalse(session.calls[0][2]["allow_redirects"])

    def test_invalid_empty_template_filter_types_are_not_silently_dropped(self):
        for filters in ([], False, 0):
            with self.subTest(filters=filters):
                client, _ = self.client(Response({"templates": [{
                    "hash_id": DEFAULT_TEMPLATE_HASH, "image": "pytorch/pytorch",
                    "tag": "2.4.0-cuda12.4-cudnn9-devel", "recommended_disk_space": 32,
                    "extra_filters": filters,
                }]}))
                with self.assertRaises(JobError):
                    client.template(DEFAULT_TEMPLATE_HASH, Deadline(10))

    def test_official_cuda_version_tags_are_accepted_without_changing_the_default_pin(self):
        template_hash = "5ec457ba5e023038cdcfe7d4aab5c207"
        client, _ = self.client(Response({"templates": [{
            "hash_id": template_hash, "image": "nvidia/cuda",
            "tag": "12.4.1-devel-ubuntu22.04", "recommended_disk_space": 16,
            "extra_filters": {"cpu_arch": {"in": ["amd64", "arm64"]}},
        }]}))
        template = client.template(template_hash, Deadline(10))
        self.assertEqual(template.image, "nvidia/cuda:12.4.1-devel-ubuntu22.04")
        self.assertEqual(template.cuda_version, 12.4)
        self.assertEqual(template.recommended_disk, 16)

    def test_a_numeric_tag_on_an_unrelated_repository_is_not_a_cuda_template(self):
        template_hash = "5ec457ba5e023038cdcfe7d4aab5c207"
        client, _ = self.client(Response({"templates": [{
            "hash_id": template_hash, "image": "unrelated/image",
            "tag": "12.4.1-devel-ubuntu22.04", "recommended_disk_space": 16,
        }]}))
        with self.assertRaisesRegex(JobError, "CUDA development template"):
            client.template(template_hash, Deadline(10))

    def test_full_pages_are_followed_even_when_truncated_is_false(self):
        client, session = self.client(
            Response({"offers": [raw_offer(1, gpu_name="RTX 3090"), raw_offer(2, gpu_name="RTX 3090")],
                      "truncated": False}),
            Response({"offers": [raw_offer(3)], "truncated": False}),
        )
        with patch("vast_client.PAGE_SIZE", 2):
            offers = client.offers(TEMPLATE, 64, 4, 11100, Deadline(10))
        self.assertEqual([offer.id for offer in offers], [1, 2, 3])
        self.assertEqual(choose_offer(offers, "prefer-a100", 4).id, 3)
        query = session.calls[1][2]["json"]
        self.assertEqual(query["ask_contract_id"], {"gt": 2})
        self.assertEqual(query["order"], [["ask_contract_id", "asc"]])
        self.assertEqual(query["allocated_storage"], 64)
        self.assertEqual(query["duration"], {"gte": 11100})
        self.assertEqual(query["gpu_ram"], {"gte": 24000})
        self.assertEqual(query["type"], "ondemand")

    def test_selected_offer_is_revalidated_with_ask_contract_identity(self):
        client, session = self.client(Response({"offers": [raw_offer(9)], "truncated": False}))
        offers = client.offers(TEMPLATE, 64, 4, 11100, Deadline(10), offer_id=9)
        query = session.calls[0][2]["json"]
        self.assertEqual(query["ask_contract_id"], {"eq": 9})
        self.assertNotIn("id", query)
        self.assertEqual([offer.id for offer in offers], [9])

    def test_multi_gpu_filter_uses_exact_count_and_per_gpu_memory(self):
        client, session = self.client(Response({"offers": [
            raw_offer(1, num_gpus=4, dph_total=3),
            raw_offer(2, num_gpus=8, dph_total=7.5),
        ]}))
        offers = client.offers(TEMPLATE, 64, 8, 11100, Deadline(10), gpu_count=8)
        self.assertEqual(session.calls[0][2]["json"]["num_gpus"], {"eq": 8})
        self.assertEqual([offer.id for offer in offers], [2])
        self.assertEqual(offers[0].gpu_count, 8)
        self.assertEqual(offers[0].gpu_ram_mb, 40960)
        self.assertEqual(offers[0].summary()["total_gpu_ram_mb"], 327680)

    def test_server_filter_violation_does_not_rent_an_ineligible_gpu(self):
        cases = (
            {"gpu_ram": 23999}, {"dph_total": 4.0001}, {"num_gpus": 2},
            {"verification": "unverified"}, {"gpu_arch": "amd"}, {"cuda_max_good": 12.2},
            {"duration": 11099}, {"disk_space": 63}, {"compute_cap": 1200},
            {"verification": "unverified", "verified": True},
            {"verification": "verified", "verified": False},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                client, _ = self.client(Response({"offers": [raw_offer(1, **overrides)], "truncated": False}))
                self.assertEqual(client.offers(TEMPLATE, 64, 4, 11100, Deadline(10)), [])

    def test_api_price_and_vram_boundaries_use_total_price_including_storage(self):
        client, _ = self.client(Response({"offers": [
            raw_offer(1, gpu_ram=24000, dph_total=4, dph_base=3.5),
            raw_offer(2, dph_total=4.2, dph_base=3.8),
        ]}))
        offers = client.offers(TEMPLATE, 64, 4, 11100, Deadline(10))
        self.assertEqual([offer.id for offer in offers], [1])
        self.assertEqual(offers[0].hourly_price, 4)

    def test_malformed_offer_names_or_ports_are_not_rented(self):
        for fields in ({"gpu_name": None}, {"direct_port_count": -1}, {"direct_port_count": 0.5}):
            with self.subTest(fields=fields):
                client, _ = self.client(Response({"offers": [raw_offer(1, **fields)]}))
                with self.assertRaises(JobError):
                    client.offers(TEMPLATE, 64, 4, 11100, Deadline(10))

    def test_nonadvancing_pagination_and_missing_fields_are_errors(self):
        client, _ = self.client(
            Response({"offers": [raw_offer(1)], "truncated": True}),
            Response({"offers": [raw_offer(1)], "truncated": True}),
        )
        with self.assertRaises(JobError):
            client.offers(TEMPLATE, 64, 4, 11100, Deadline(10))
        client, _ = self.client(Response({}))
        with self.assertRaises(JobError):
            client.offers(TEMPLATE, 64, 4, 11100, Deadline(10))

    def test_creation_returns_contract_id_and_never_forwards_credentials(self):
        client, session = self.client(Response({"success": True, "new_contract": 999}))
        result = client.create(fixture_offer(12), TEMPLATE, 64, "run-label", Deadline(10))
        self.assertEqual(result, 999)
        method, url, request = session.calls[0]
        self.assertEqual(method, "PUT")
        self.assertTrue(url.endswith("/asks/12/"))
        self.assertEqual(request["json"]["runtype"], "ssh_direct")
        self.assertEqual(request["json"]["template_hash_id"], DEFAULT_TEMPLATE_HASH)
        self.assertEqual(request["json"]["image"], DEFAULT_IMAGE)
        self.assertFalse(request["json"]["vm"])
        self.assertNotIn(VAST_TOKEN, json.dumps(request))
        self.assertNotIn("price", request["json"])

    def test_lost_creation_response_is_ambiguous_and_not_retried(self):
        client, session = self.client(requests.ConnectionError("private url"))
        with self.assertRaises(ApiError) as caught:
            client.create(fixture_offer(), TEMPLATE, 64, "label", Deadline(10))
        self.assertTrue(caught.exception.ambiguous)
        self.assertEqual(len(session.calls), 1)

    def test_invalid_success_response_is_not_allocation_success(self):
        for data in ({"success": True, "id": 9}, {"success": True, "new_contract": True},
                     {"new_contract": 9}):
            with self.subTest(data=data):
                client, _ = self.client(Response(data))
                with self.assertRaises(ApiError) as caught:
                    client.create(fixture_offer(), TEMPLATE, 64, "label", Deadline(10))
                self.assertTrue(caught.exception.ambiguous)

    def test_account_keys_are_never_registered(self):
        client, session = self.client(Response({"success": True}))
        client.attach_key(999, "ssh-ed25519 TEST_PUBLIC_KEY", Deadline(10))
        self.assertEqual(session.calls[0][0], "POST")
        self.assertTrue(session.calls[0][1].endswith("/instances/999/ssh/"))
        self.assertEqual(session.calls[0][2]["json"], {"ssh_key": "ssh-ed25519 TEST_PUBLIC_KEY"})

    def test_lookup_and_destruction_accept_confirmed_absence(self):
        client, session = self.client(Response({"instances": None}), Response({}, status=404))
        self.assertIsNone(client.instance(123, Deadline(10)))
        client.destroy(123, Deadline(10))
        self.assertEqual(session.calls[1][0], "DELETE")

    def test_reconciliation_uses_owned_label_and_v1_pagination(self):
        client, session = self.client(
            Response({"instances": [{"id": 1, "label": "label"}], "next_token": "next"}),
            Response({"instances": [{"id": 2, "label": "label"}], "next_token": None}),
        )
        self.assertEqual([row["id"] for row in client.labeled_instances("label", Deadline(10))], [1, 2])
        self.assertTrue(session.calls[0][1].endswith("/api/v1/instances/"))
        self.assertEqual(json.loads(session.calls[0][2]["params"]["select_filters"]),
                         {"label": {"eq": "label"}})
        self.assertEqual(session.calls[1][2]["params"]["after_token"], "next")

    def test_reconciliation_refuses_unrelated_instances(self):
        client, _ = self.client(Response({"instances": [{"id": 1, "label": "not-owned"}]}))
        with self.assertRaises(JobError):
            client.labeled_instances("owned", Deadline(10))

    def test_read_retries_are_bounded_and_errors_are_redacted(self):
        client, session = self.client(
            Response({"msg": "retry"}, 429),
            Response({"msg": "retry"}, 503),
            Response({"msg": f"denied {VAST_TOKEN} https://unit.invalid/api?key={VAST_TOKEN}"}, 401),
        )
        clock = Clock()
        with self.assertRaises(ApiError) as caught:
            client.instance(1, clock.deadline(30))
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(clock.value, 3)
        self.assertNotIn(VAST_TOKEN, str(caught.exception))

    def test_http_timeouts_cannot_exceed_the_remaining_budget(self):
        client, session = self.client(Response({"instances": None}))
        client.instance(1, Clock().deadline(7))
        connect, read = session.calls[0][2]["timeout"]
        self.assertEqual(connect + read, 7)

    def test_invalid_json_on_transient_http_errors_is_retried(self):
        client, session = self.client(
            Response(ValueError("not JSON"), 503), Response({"instances": None}),
        )
        clock = Clock()
        self.assertIsNone(client.instance(1, clock.deadline(30)))
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(clock.value, 1)

    def test_invalid_json_creation_failure_is_not_retried(self):
        client, session = self.client(Response(ValueError("not JSON"), 503))
        with self.assertRaises(ApiError) as caught:
            client.create(fixture_offer(), TEMPLATE, 64, "label", Deadline(10))
        self.assertTrue(caught.exception.ambiguous)
        self.assertEqual(len(session.calls), 1)
