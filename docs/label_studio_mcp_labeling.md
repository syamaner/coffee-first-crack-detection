# Coffee First Crack Labeling Runbook

## 1. What you will label

Label only the 38 staged mic1 recordings in:

```text
/Users/sertanyamaner/git/coffee-first-crack-detection/data/raw/mcp-captures/mic1/
```

Important:

- There must be exactly 38 WAV files.
- Do not label anything from `mcp-captures/mic2/`.
- Do not rename the UUID-prefixed files.
- Do not import files directly from `/Users/sertanyamaner/roasts/captures`.
- Keep the Label Studio project local because recordings may contain ambient conversation.

Optional count check:

```bash
find /Users/sertanyamaner/git/coffee-first-crack-detection/data/raw/mcp-captures/mic1 \
  -maxdepth 1 -type f -name '*.wav' | wc -l
```

Expected result:

```text
38
```

## 2. Start Label Studio

From a terminal:

```bash
label-studio start
```

Open:

```text
http://localhost:8080
```

Sign in to your existing local Label Studio account if prompted.

## 3. Create the project

1. Select **Create Project** or **New Project**.
2. Name it:

   ```text
   Coffee First Crack — MCP Mic1 2026-08
   ```

3. Do not import data yet if Label Studio first asks you to configure the labeling interface.

## 4. Configure the labeling interface

1. Open **Settings → Labeling Interface**.
2. Choose **Audio/Speech Processing → Audio Classification with Regions**.
3. Replace the configuration with:

```xml
<View>
  <Audio name="audio" value="$audio" zoom="true" waveformHeight="100"/>
  <Labels name="label" toName="audio">
    <Label value="first_crack" background="#FF0000"/>
  </Labels>
</View>
```

4. Save the interface.

Only one label is permitted: `first_crack`.

Everything outside the selected region is automatically treated as `no_first_crack`.

## 5. Import the recordings

1. Open the project's **Import** page.
2. Choose **Upload Files**.
3. Navigate to:

   ```text
   /Users/sertanyamaner/git/coffee-first-crack-detection/data/raw/mcp-captures/mic1/
   ```

4. Select all 38 WAV files.
5. Upload/import them.
6. Confirm that the project shows exactly **38 tasks**.

Stop and report the discrepancy if the task count is not 38. Do not compensate by importing
mic2 files.

### Recover from an accidental duplicate import

Task deletion is permanent, so use the Label Studio UI and never edit its database directly.

- If labeling has not started, select all 76 tasks in **Data Manager**, choose **Delete tasks**
  from the selected-task dropdown, confirm, and import the 38 mic1 WAVs once.
- If labeling has started, export a JSON backup, sort **Data Manager** by task ID, verify the
  newer 38 copies are unlabeled and match the older filenames, then delete only those newer
  tasks.
- Confirm exactly 38 tasks remain before continuing. Choose **Delete tasks**, not **Delete
  annotations**, and do not delete the project or staged WAVs.

## 6. Label each recording

For every task:

1. Open the task.
2. Play the recording and navigate toward the expected first-crack period.
3. Listen for the first genuine first-crack pop.
4. Zoom in around that point.
5. Select the red `first_crack` label.
6. Draw exactly one continuous region:
   - Start: the first credible first-crack pop.
   - End: the point where consistent first-crack activity has finished.
7. Submit the task.
8. Continue to the next recording.

The region may last several minutes. Do not draw a separate region around every pop.

### What belongs inside the region

Include:

- The first credible pop.
- Sparse early pops immediately following it.
- The sustained first-crack period.
- The fading tail until consistent cracking has ended.

There is no fixed 30-second cutoff after first crack begins. If isolated pops occur only after
an approximately 30-second quiet gap following the main cluster, end the region at the last pop
of the preceding consistent activity and exclude those isolated tail pops. If cracking resumes
as a genuine cluster rather than one or two sporadic pops, include the renewed cluster.

Do not include:

- Charge, drying, or browning sounds before first crack.
- Fan, drum, bean movement, or general mechanical noise by itself.
- Second crack.
- Cooling-tray sounds after drop.
- Isolated noises that do not sound like coffee cracking.

## 7. Handle uncertain recordings

If the first pop is difficult to locate:

1. Replay the surrounding section.
2. Zoom into the waveform.
3. Compare suspected pops with nearby mechanical or handling noise.
4. Prefer a slightly wider boundary when uncertainty is only a few seconds.

Do not create several alternative regions. Choose the best single boundary.

If a recording genuinely contains no detectable first crack:

1. Draw no region.
2. Submit the task with zero regions.

Do not invent a region merely to make every task positive.

## 8. Quality check before export

Confirm:

- All 38 tasks are submitted, not drafts.
- Every task has either zero or one region.
- Every region is labeled exactly `first_crack`.
- No mic2 recording was imported.
- No individual pops were labeled as separate regions.
- The beginning marks the first credible pop.
- The end marks the end of consistent first-crack activity.

The conversion pipeline will reject duplicate tasks, unsupported labels, multiple regions, or
invalid time boundaries.

## 9. Export the annotations

1. Return to the project overview.
2. Select **Export**.
3. Choose **JSON** as the export format.
4. Download the export.

Do not export CSV, JSON-MIN, or another format.

## 10. Place the export in the repository

Create the destination directory:

```bash
mkdir -p /Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/mcp
```

Move the downloaded JSON to:

```text
/Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/mcp/labelstudio-export.json
```

For example:

```bash
mv "/path/to/downloaded/project-export.json" \
  /Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/mcp/labelstudio-export.json
```

Verify it exists:

```bash
test -f \
  /Users/sertanyamaner/git/coffee-first-crack-detection/data/labels/mcp/labelstudio-export.json \
  && echo "Label Studio export ready"
```

Expected output:

```text
Label Studio export ready
```

Do not commit this export or modify Label Studio's internal database.

Once the file is present, tell Codex: **“The Label Studio export is ready.”** Codex can then
continue with validation, mic2 derivation, pair-safe rebuild, training, evaluation, and ONNX
comparison.
