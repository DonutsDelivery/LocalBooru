; LocalBooru Portable Self-Extractor
; On first run: extracts app files to LocalBooru\ subfolder, launches app,
; then replaces itself with a tiny stub for instant subsequent launches.
;
; Build with:
;   makensis /DVERSION=x.y.z /DAPP_FILES=path\to\unpacked /DICON_FILE=path\to\icon.ico portable-launcher.nsi

!ifndef VERSION
    !define VERSION "0.0.0"
!endif

!ifndef ICON_FILE
    !define ICON_FILE "..\assets\icon.ico"
!endif

Name "LocalBooru ${VERSION} Portable"
OutFile "..\dist\LocalBooru-Portable.exe"
Icon "${ICON_FILE}"
RequestExecutionLevel user
SilentInstall silent
SetCompressor /SOLID lzma

Section
    ; Target directory for extracted files
    StrCpy $INSTDIR "$EXEDIR\LocalBooru"

    ; --- Check if already extracted with matching version ---
    IfFileExists "$INSTDIR\.portable-version" 0 extract

    ; Read stored version
    FileOpen $1 "$INSTDIR\.portable-version" r
    FileRead $1 $2
    FileClose $1

    ; Compare - if match, skip extraction
    StrCmp $2 "${VERSION}" launch

extract:
    ; Extract all files to LocalBooru\ subfolder
    SetOutPath $INSTDIR
    File /r "${APP_FILES}\*.*"

    ; Write version marker
    FileOpen $1 "$INSTDIR\.portable-version" w
    FileWrite $1 "${VERSION}"
    FileClose $1

    ; --- Self-replacement: create a .cmd that waits then overwrites this exe with the stub ---
    ; We write the resolved paths directly into the script
    StrCpy $3 "$EXEDIR\.replace-launcher.cmd"
    FileOpen $1 $3 w
    FileWrite $1 "@echo off$\r$\n"
    ; Wait for NSIS exe to release its file lock
    FileWrite $1 "ping -n 4 127.0.0.1 >nul$\r$\n"
    ; Copy stub over the big exe
    FileWrite $1 'copy /y "$INSTDIR\.portable-stub.exe" "$EXEPATH"$\r$\n'
    ; Clean up this script
    FileWrite $1 '(goto) 2>nul & del "%~f0"$\r$\n'
    FileClose $1

    ; Launch the replacement script minimized (fire-and-forget)
    Exec '"cmd.exe" /c start /min "" "$3"'

launch:
    ; Launch the actual app
    IfFileExists "$INSTDIR\LocalBooru.exe" 0 error
    Exec '"$INSTDIR\LocalBooru.exe"'
    Goto done

error:
    MessageBox MB_OK|MB_ICONERROR "LocalBooru.exe not found after extraction.$\r$\n$\r$\nExpected: $INSTDIR\LocalBooru.exe$\r$\n$\r$\nThe archive may be corrupted. Please re-download."

done:
SectionEnd
