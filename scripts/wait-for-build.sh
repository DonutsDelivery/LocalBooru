#!/bin/bash
# Wait for GitHub Actions build to complete and play a sound

REPO="DonutsDelivery/LocalBooru"
SOUND_SUCCESS="/usr/share/sounds/freedesktop/stereo/complete.oga"
SOUND_FAIL="/usr/share/sounds/freedesktop/stereo/dialog-error.oga"

# Get the latest run ID if not provided
RUN_ID=${1:-$(gh run list -R "$REPO" --limit 1 --json databaseId -q '.[0].databaseId')}

echo "Watching build $RUN_ID..."
echo "https://github.com/$REPO/actions/runs/$RUN_ID"

while true; do
    STATUS=$(gh run view "$RUN_ID" -R "$REPO" --json status,conclusion -q '.status + " " + (.conclusion // "")')

    if [[ "$STATUS" == "completed success" ]]; then
        echo "✅ Build succeeded!"
        paplay "$SOUND_SUCCESS" 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || echo -e "\a"

        # Show release size
        VERSION=$(gh run view "$RUN_ID" -R "$REPO" --json headBranch -q '.headBranch' | grep -oE 'v[0-9.]+' || echo "")
        if [[ -n "$VERSION" ]]; then
            echo "Release assets:"
            gh release view "$VERSION" -R "$REPO" --json assets -q '.assets[] | "\(.name): \(.size / 1048576 | floor)MB"' 2>/dev/null
        fi
        exit 0

    elif [[ "$STATUS" == "completed failure" ]] || [[ "$STATUS" == "completed cancelled" ]]; then
        echo "❌ Build failed!"
        paplay "$SOUND_FAIL" 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null || echo -e "\a"
        exit 1
    fi

    echo -n "."
    sleep 10
done
