# Gold graded 0–3 mapping

**Không gọi ground truth.** Đây là *human-annotated graded evaluation set*.

## Files

| File | Vai trò |
|---|---|
| `human_validated_benchmark.csv` | Bản gốc Phase 3, `human_relevance` ∈ {0,1,2} |
| `human_validated_benchmark_graded_0_3.csv` | Bản đánh giá paper, cột `relevance` ∈ {0,1,2,3} |

## Rubric (DE_CUONG_CHOT §11.1)

| relevance | Ý nghĩa |
|---|---|
| 3 | Highly suitable |
| 2 | Suitable but missing some requirements |
| 1 | Weak relevance |
| 0 | Not suitable |

## Mapping có kiểm soát

Từ `human_relevance` (0–2) + aspect scores (`human_role_score`, `human_skill_score`, `human_exp_score`, `human_loc_score`, mỗi cái 0–2):

| Điều kiện | → relevance | `mapping_rule` |
|---|---|---|
| `human_relevance == 0` | 0 | `rel0->0` |
| `human_relevance == 1` | 1 | `rel1->1` |
| `human_relevance == 2` và `min(role,skill,exp,loc) >= 2` | 3 | `rel2+all_aspects>=2->3` |
| `human_relevance == 2` còn gap aspect | 2 | `rel2+gap->2` |

## Phân phối sau map (n=100)

| relevance | n |
|---|---|
| 0 | 30 |
| 1 | 35 |
| 2 | 29 |
| 3 | 6 |

Một annotator; không có IAA. Số grade=3 thấp vì chỉ tách từ lớp “2” khi mọi aspect đều tối đa.
