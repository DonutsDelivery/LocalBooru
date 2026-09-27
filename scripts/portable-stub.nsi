; DonutMediaCenter Portable Stub Launcher
; Tiny exe (~50KB) that just launches DonutMediaCenter\DonutMediaCenter.exe relative to itself.
; After first extraction, the big self-extractor replaces itself with this stub
; so subsequent launches are instant.

Name "DonutMediaCenter Portable"
OutFile "..\dist\portable-stub.exe"
Icon "..\assets\icon.ico"
RequestExecutionLevel user
SilentInstall silent

Section
    ; $EXEDIR = directory where this exe lives
    ; DonutMediaCenter.exe is in DonutMediaCenter\ subfolder next to this exe
    StrCpy $0 "$EXEDIR\DonutMediaCenter\DonutMediaCenter.exe"

    IfFileExists $0 launch error

launch:
    Exec '"$0"'
    Goto done

error:
    MessageBox MB_OK|MB_ICONSTOP "DonutMediaCenter not found. Please re-download DonutMediaCenter-Portable.exe."

done:
SectionEnd
