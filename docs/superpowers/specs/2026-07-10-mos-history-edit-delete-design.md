# MOS History Edit and Delete Design

## Scope

Every comparison shown in MOS history can be changed or deleted. The history remains paginated and virtualized. This change does not alter pair sampling, new-rating submission, training manifests, model training, or inference.

## User interface

Each history row always shows `Change` and `Delete` actions.

- `Change` opens the existing inline score editor and retains the two combined `Choose A/B and save` actions.
- `Delete` immediately calls the delete API. It does not use `window.confirm`, `window.alert`, `window.prompt`, or another browser-native dialog.
- Pending mutations disable row actions. Success and failure continue to use the existing in-app toast feedback.
- The former `Undo` wording is removed because the action is a literal history-record deletion.

## API and persistence

`PATCH /mos/ratings/{comparison_id}` and `DELETE /mos/ratings/{comparison_id}` accept any existing comparison. The newest-only restriction is removed. Missing comparisons continue to return an actionable client error.

The audio file score represents the newest remaining MOS comparison involving that audio. Mutation behavior preserves this invariant:

1. Changing an older comparison updates that history record but does not replace a score supplied by a newer comparison.
2. Changing the newest comparison involving an audio updates that audio's current score.
3. Deleting an older comparison leaves current scores from newer comparisons unchanged.
4. Deleting the newest comparison involving an audio restores the score from the previous remaining comparison.
5. If no comparison remains for that audio, deletion restores the score that existed before the deleted comparison chain began.

The existing `previous_score_a` and `previous_score_b` fields form the rollback chain. When an older comparison is changed or deleted, the next comparison involving each affected audio is rewired so its previous score remains correct. This avoids adding another table or migration.

History responses keep the existing `can_modify` field for API compatibility, but it is `true` for every row.

## Error handling

The backend validates that the preferred audio belongs to the comparison. Invalid or missing comparison IDs fail clearly. Database changes to the comparison, score chain, and current audio scores occur in one transaction.

## Verification

Temporary tests cover:

- changing an older comparison without clobbering a newer current score;
- changing the latest comparison and updating the current score;
- deleting an older comparison and rewiring the next comparison's rollback value;
- deleting comparisons newest-to-oldest and restoring the pre-MOS score;
- every history row being modifiable;
- the MOS frontend containing `Delete` and no browser-native dialog calls.

The tests are removed before completion per repository policy. The frontend production build, Python compilation, OpenAPI export, and a live API smoke flow run through the Nix development shell.
