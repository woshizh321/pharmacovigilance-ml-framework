# Code Provenance

The release contains selected repaired/final scripts copied from the frozen PDS project. No scientific data or generated result artifact is included. Release copies were mechanically sanitized only to remove local absolute path prefixes; source scripts were not modified.

| Script | Frozen source SHA-256 | Release SHA-256 | Release transformation |
|---|---|---|---|
| `scripts/p08_jader_v5_repreflight.py` | `cbffc619e394a2f9691ab9956ceed391f1b106fa0f7744baf72427e2b5cbf1d3` | `443fa0ed571959002ba4949f12a604b4d23e4e6253e1bfcc7136f3f56f270c14` | local path prefixes replaced |
| `scripts/p10_faers_meddra28_repair.py` | `f5e9f483d3eac981c3dd085628573c085dc56a9e04a7c68b7ecc422638d18191` | `089a57463b4e86aed1a820b23f65f2635e5b82bfd6b48facfc614be40d887762` | local path prefixes replaced |
| `scripts/p11_rebuild_faers_case_pt.py` | `e9abfbf94b2d1d5f3d5ec25f01e4b81199f0a88f191d9af6fcb1b264cdc92c3f` | `dc9d4656e0bba3b39de0a786f38f05caf59110cabc88dd9917e2da1f17b46a75` | local path prefixes replaced |
| `scripts/p12_rebuild_aact_meddra28_ceiling.py` | `30ffe7b6f8132fe3d6545f79d0b2426dcda81fe72a01390253186ccedc9d7cf1` | `0d38d7357c63960859d28ff88cc597d7a5be437161c647c0083fef55d7420596` | local path prefixes replaced |
| `scripts/p13_finalize_faers_pt_repair_hold.py` | `f4de8ee59e7355b59f3fc1faf11a35165d9c49ea9c15ba5f76c902d78ae1effa` | `3690cccd2ea0537f60c11ef6b917cf1e940772b1918c177ffee2d88d3825117d` | local path prefixes replaced |
| `scripts/p14_build_fda_regulatory_cohort.py` | `99c0e834da3f8b538a348be8f370ec25992291ef570c7024d8781c44937a3373` | `ed2a546e0c2471f2bea32de02495aedfedde02754274166a853759c4025011ad` | local path prefixes replaced |
| `scripts/p15_build_drug_identity_master.py` | `3ba29940430a889d4ba252ddaa46c14971dee9c2d3686e1cee078fc2d7c4d1f0` | `6bb1786cf3a51e9a31a870f8ec192224101e8e9d39b74e814f4eed7ad92b8c9a` | local path prefixes replaced |
| `scripts/p16_build_bstrict_and_arm_audit.py` | `18be2652a5cba678305af8ce3b2c126e7c49c4b2110b7a3ebad87c40012671cb` | `440a0e9350c69a5c24ee17c742f980691d48e83d79f48da38e50c7bd6fcc44bb` | local path prefixes replaced |
| `scripts/p17_build_fda_anchored_faers_labels.py` | `62afedc12cf2dd8ba8daeb711176b51085b5aafa8421518df63d4350a49949be` | `a2b25469827e6ec635cfe18757ef181c9a58ba12161d8189077b7808f1b90ebf` | local path prefixes replaced |
| `scripts/p18_assessability_coverage_splits.py` | `8979f31561d5e04fee9ee6b07c160131b13c57eb8e435276b29415b0c4be1e25` | `f46acb3f41591a86fb18c1ad18b710c0739c45c097fd889d217356c07ea6a999` | local path prefixes replaced |
| `scripts/p19_jader_v5_replication_rebuild.py` | `7ca8822b7750699915d885849418f1d61f85b635f73cac6cd440d5f1a07adef3` | `59993e754d39f65b6f5947bf33db90a701fddc58d6653e9a2ada33e5f93a421a` | local path prefixes replaced |
| `scripts/p20_close_preflight_v2.py` | `a43f5b6a5cd19b0fbdbaecebb0bd5233d957ba8d4b1914372ec8cfa5e6d97112` | `c5ffa4d6f81ae93ff00baa1f73c6e0da8aa8693263f1eade06da6f5225b6610b` | local path prefixes replaced |
| `scripts/s01_section1_analysis.py` | `bcf0b53de88afccf52212a0d866e969f39310e4855a7412739097f2098b857e8` | `5fa677ec9f4b61e158d25f89165ceac2bb5b4984dba7a6cf11fb1786e23632bb` | local path prefixes replaced |
| `scripts/s02_command07_targeted_amendment.py` | `5399c8982f40e6de1ebedeff06f92ad82e94c50d73ec0c6a9724d4aab027e700` | `c1269c75bca099f1dab597a0e10c023e80012e12fe0e8a5e863edb11960a95de` | local path prefixes replaced |
| `scripts/s02_section2_coverage.py` | `8a2c65aa3b8effe026f180ebc9a113108e90d8f774dd47e2df7c6fcd97f948f0` | `d41e51921eb85522273b3ad2b4f70b359e866af9b4572eb4145c0701869666e2` | local path prefixes replaced |
| `scripts/s03a_feature_matrix_and_protocol.py` | `954e633fa607247850f3fa3cce26b6ac48797228bf68159309b4a75a15c7b360` | `e1202ac6f9f42cf7deaeb8eee524c976ad992594a77b4673b1b5788024afe663` | local path prefixes replaced |
| `scripts/s03b_finalize_from_saved_outputs.py` | `90d4bd987ec71a4091af11c0a1801b7ad5c9c0adf0301213fa94d2039179e951` | `90d4bd987ec71a4091af11c0a1801b7ad5c9c0adf0301213fa94d2039179e951` | identical |
| `scripts/s03b_nested_training_and_freeze.py` | `f0a045ed0ba67860476a9416bbd61a7bc89cb47cc733e5f6b112a54e44eae99a` | `f0a045ed0ba67860476a9416bbd61a7bc89cb47cc733e5f6b112a54e44eae99a` | identical |
| `scripts/s03c_preholdout_lock.py` | `020a3f1ea7cc8ebd829a2bac04d16df838839e456bdf10fb614bd7445781a8e3` | `020a3f1ea7cc8ebd829a2bac04d16df838839e456bdf10fb614bd7445781a8e3` | identical |
| `scripts/s04a_holdout_feature_scoring.py` | `7f811a7f714771aba18814265eced71d2696258f058b931844424e8132ceab47` | `025aaf930ec4eface529e47bf972d02299b834530e02f728ead05432544d7baa` | local path prefixes replaced |
| `scripts/s04b_holdout_outcome_evaluation.py` | `5ad9f16d4216d300b68f43269f7075e3d913f9e9bee80b9e50ef8c8b4654cd7e` | `5ad9f16d4216d300b68f43269f7075e3d913f9e9bee80b9e50ef8c8b4654cd7e` | identical |
| `scripts/s04c_finalize_section4_qc.py` | `76cbe5cc3682d394fcc68335b03f92c07d9fb8ee44c2261fda5e2683acfcae2b` | `76cbe5cc3682d394fcc68335b03f92c07d9fb8ee44c2261fda5e2683acfcae2b` | identical |
| `scripts/s04d_audit_section4_outputs.py` | `4e59dfdde3bea8cdc1843590575039c7eba596cf633a2b23090edb001c166d89` | `4e59dfdde3bea8cdc1843590575039c7eba596cf633a2b23090edb001c166d89` | identical |
| `scripts/s05_audit_interpretation_outputs.py` | `947e38a669467605dc7efbff22827f8bf34a1fdbb3f82b077afb87be701d5eb4` | `947e38a669467605dc7efbff22827f8bf34a1fdbb3f82b077afb87be701d5eb4` | identical |
| `scripts/s05_cross_model_interpretation.py` | `0db100628daddf13c709a19e282aaa12db80b6936e7fa2c070a49d6090206531` | `0db100628daddf13c709a19e282aaa12db80b6936e7fa2c070a49d6090206531` | identical |
| `scripts/s06_audit_robustness_outputs.py` | `0a11358b700a9a2a9ed9a9e6d4eeab93d953b68502bccdb60ca683f420e096f6` | `23f61b386c02656b821677cef6885c69235890203714057066a9eab9c0559a9f` | local path prefixes replaced |
| `scripts/s06_cross_database_robustness.py` | `9404b476c8b578169608e6686b9218fcd79b6f8ef9c39c2c2e4da8bab1d72172` | `7c5a304bf75fe5cf454e4d5af6e30624b719f05047a951cd7f88daf67fe3e894` | local path prefixes replaced |
| `scripts/s17_publication_assets.py` | `67695c1402934e4a9bd8afabd1be8d732defc00c014c9b28029e40d6b3253c56` | `67695c1402934e4a9bd8afabd1be8d732defc00c014c9b28029e40d6b3253c56` | identical |
| `scripts/s17_visualization_qc.py` | `4663d0d2e1787ddfcbce1be42f3a647aaf6c327b1a6770f9be79fb84e60945af` | `4663d0d2e1787ddfcbce1be42f3a647aaf6c327b1a6770f9be79fb84e60945af` | identical |
| `scripts/s19_table_qc.py` | `4b47ef056545a4bd21f634ee4637d59c9628000338bb519e5c99e03893d9b8d4` | `4b47ef056545a4bd21f634ee4637d59c9628000338bb519e5c99e03893d9b8d4` | identical |
| `scripts/s20_visual_lock.py` | `eb0d884bcfd11b05629be9a8538d8f5825572dd2aa8729c3ac860e3eec3bbcf8` | `eb0d884bcfd11b05629be9a8538d8f5825572dd2aa8729c3ac860e3eec3bbcf8` | identical |

- Included scripts: 31.
- Omitted preliminary/superseded scripts: `p01`–`p07` and the JADER-v4 delta/gate script.
- Omitted manuscript/reference/reporting QC scripts: they depend on internal publication artifacts not included in this release.
- A differing release hash indicates path-literal sanitization, not an intentional algorithmic change.
- The original source project remains the authority for scientific provenance.
