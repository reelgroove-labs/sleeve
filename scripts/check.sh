#!/bin/zsh
set -eu
cd "${0:A:h}/.."
ROOT_DIR="$PWD"
CHECK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sleeve-checks.XXXXXX")"
RUNTIME_PY="$HOME/Library/Application Support/com.repriselabs.Sleeve/Runtime/bin/python3"
if [[ ! -x "$RUNTIME_PY" ]]; then RUNTIME_PY=/usr/bin/python3; fi
export PYTHONDONTWRITEBYTECODE=1
"$RUNTIME_PY" Tests/check_names.py
"$RUNTIME_PY" Tests/check_placement.py
"$RUNTIME_PY" Tests/check_arrange.py
"$RUNTIME_PY" Tests/check_scan_order.py
"$RUNTIME_PY" Tests/check_edge_line.py
"$RUNTIME_PY" Tests/check_paper_edge.py
"$RUNTIME_PY" Tests/check_label_side_cold.py
"$RUNTIME_PY" Tests/check_label_tight_scan.py
"$RUNTIME_PY" Tests/check_label_texture_edge.py
"$RUNTIME_PY" Tests/check_vakken_refine.py
"$RUNTIME_PY" Tests/check_derived_files.py
"$RUNTIME_PY" Tests/check_insert_sheets.py
"$RUNTIME_PY" Tests/check_barcode_read.py
"$RUNTIME_PY" Tests/check_upright_numbering.py
"$RUNTIME_PY" Tests/check_disc_loose_edge.py
"$RUNTIME_PY" Tests/check_colour_panel.py
"$RUNTIME_PY" Tests/check_icc.py
for check in bridge audits progress cancel circle_edit memory scratch scratch_work release_advice blend fields handoff preview_cache workers facts_fixes source_checks label_material label_orientation paper_detail; do
 "$RUNTIME_PY" "Tests/check_${check}.py" "$CHECK_DIR/$check"
done
mkdir -p "$CHECK_DIR/Checks.app/Contents/MacOS"
ln -s "$ROOT_DIR/Resources" "$CHECK_DIR/Checks.app/Contents/Resources"
cat > "$CHECK_DIR/Checks.app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>CFBundleExecutable</key><string>Checks</string><key>CFBundleIdentifier</key><string>com.repriselabs.sleeve.checks</string><key>CFBundlePackageType</key><string>APPL</string></dict></plist>
PLIST
/usr/bin/swiftc -swift-version 5 -O -parse-as-library Source/StudioModel.swift Source/StudioProjects.swift Source/StudioEditing.swift Source/StudioDiscovery.swift Source/StudioColour.swift Source/StudioArrange.swift Source/StudioRelease.swift Tests/ProjectQueueChecks.swift -o "$CHECK_DIR/Checks.app/Contents/MacOS/Checks" -framework SwiftUI -framework AppKit
"$CHECK_DIR/Checks.app/Contents/MacOS/Checks" "$CHECK_DIR/project-queue"
/usr/bin/swiftc -swift-version 5 -O -parse-as-library Source/StudioModel.swift Source/StudioProjects.swift Source/StudioEditing.swift Source/StudioDiscovery.swift Source/StudioColour.swift Source/StudioArrange.swift Source/StudioRelease.swift Tests/EditingChecks.swift -o "$CHECK_DIR/Checks.app/Contents/MacOS/Checks" -framework SwiftUI -framework AppKit
"$CHECK_DIR/Checks.app/Contents/MacOS/Checks" "$CHECK_DIR/editing"
/usr/bin/swiftc -swift-version 5 -O -parse-as-library Source/StudioModel.swift Source/StudioProjects.swift Source/StudioEditing.swift Source/StudioDiscovery.swift Source/StudioColour.swift Source/StudioArrange.swift Source/StudioRelease.swift Tests/ColourWorkflowChecks.swift -o "$CHECK_DIR/Checks.app/Contents/MacOS/Checks" -framework SwiftUI -framework AppKit
"$CHECK_DIR/Checks.app/Contents/MacOS/Checks" "$CHECK_DIR/colour-workflow"
/usr/bin/swiftc -swift-version 5 -O -parse-as-library Source/StudioModel.swift Source/StudioProjects.swift Source/StudioEditing.swift Source/StudioDiscovery.swift Source/StudioColour.swift Source/StudioArrange.swift Source/StudioRelease.swift Tests/ModelChecks.swift -o "$CHECK_DIR/Checks.app/Contents/MacOS/Checks" -framework SwiftUI -framework AppKit
"$CHECK_DIR/Checks.app/Contents/MacOS/Checks" "$CHECK_DIR/model"
print "Controles geslaagd. Verslagen: $CHECK_DIR"
