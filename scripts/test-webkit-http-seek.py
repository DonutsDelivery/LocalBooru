#!/usr/bin/env python3
"""Compile the production HTTP seek branch with real GStreamer and synthetic pads.
No WebKit app, user profile, socket, or media is opened by this fixture.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
RELATIVE = Path('Source/WebCore/platform/graphics/gstreamer/MediaPlayerPrivateGStreamer.cpp')
HEADER = r'''
#include <gst/gst.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <atomic>
#include <optional>
#undef GST_INFO_OBJECT
#define GST_INFO_OBJECT(...) ((void)0)
#define ENABLE(x) 0
#undef GST_DEBUG_OBJECT
#define GST_DEBUG_OBJECT(...) ((void)0)
namespace WTF { using std::move; }
template<class T> using Function = std::function<T>;
static GstElement* retain(GstElement* p) { return GST_ELEMENT(gst_object_ref(p)); }
static GstEvent* retain(GstEvent* p) { return gst_event_ref(p); }
static void release(GstElement* p) { gst_object_unref(p); }
static void release(GstEvent* p) { gst_event_unref(p); }
template<class T> struct GRefPtr {
    T* ptr=nullptr;
    GRefPtr()=default; GRefPtr(T* p):ptr(p) { }
    GRefPtr(const GRefPtr& other):ptr(other.ptr ? retain(other.ptr):nullptr) { }
    GRefPtr(GRefPtr&& other):ptr(std::exchange(other.ptr,nullptr)) { }
    ~GRefPtr() { if(ptr) release(ptr); }
    GRefPtr& operator=(GRefPtr&& other) { if(ptr)release(ptr); ptr=std::exchange(other.ptr,nullptr); return *this; }
    T* get() const { return ptr; } T* leakRef() { return std::exchange(ptr,nullptr); }
    explicit operator bool() const { return ptr; }
    bool operator!=(const GRefPtr& other) const { return ptr!=other.ptr; }
};
template<class T> GRefPtr<T> adoptGRef(T* p) { return GRefPtr<T>(p); }
template<class T> struct RefPtr { T* ptr; RefPtr(T* p):ptr(p) { } T* operator->()const{return ptr;} explicit operator bool()const{return ptr;} };
struct MediaTime {
    double value;
    bool isValid()const{return value>=0;} static MediaTime zeroTime(){return {0};} static MediaTime invalidTime(){return {-1};}
    bool operator==(MediaTime b)const{return value==b.value;}
    bool operator<(MediaTime b)const{return value<b.value;} bool operator<=(MediaTime b)const{return value<=b.value;}
    bool operator>=(MediaTime b)const{return value>=b.value;} bool operator!=(MediaTime b)const{return value!=b.value;}
};
static GstClockTime toGstClockTime(MediaTime time){if(time.value<0)return GST_CLOCK_TIME_NONE;return static_cast<GstClockTime>(time.value*GST_SECOND);}
struct SeekTarget {MediaTime time;};
struct MediaPlayer { enum class NetworkState { DecodeError, Empty }; enum class ReadyState { HaveNothing }; };
struct FakePlayer { bool looping=false; bool isLooping()const{return looping;} float rate()const{return 1;} void rateChanged(){ } };
struct PlayerPtr {FakePlayer value; FakePlayer* get(){return &value;} };
constexpr const char* operator""_s(const char* text,size_t){return text;}
struct URL {bool http=true,hls=false; bool protocolIsInHTTPFamily()const{return http;}
    struct Path {bool hls; bool endsWithIgnoringASCIICase(const char*)const{return hls;} };
    Path path()const{return {hls};}
};
struct GStreamerQuirksManager {
    static GStreamerQuirksManager& singleton(){static GStreamerQuirksManager q;return q;}
    std::pair<bool,bool> applyCustomInstantRateChange(bool,bool,float,bool,GstElement*){return {false,false};}
    bool isEnabled()const{return false;} void resetBufferingPercentage(void*,int) { }
};
class MediaPlayerPrivateGStreamer;
struct ThreadSafeWeakPtr {
    std::weak_ptr<MediaPlayerPrivateGStreamer> weak;
    ThreadSafeWeakPtr(MediaPlayerPrivateGStreamer&);
    auto get()const{return weak.lock();}
};
struct RunLoop {
    static RunLoop& mainSingleton(){static RunLoop loop;return loop;}
    void dispatch(std::function<void()> fn) {
        auto* task=new std::function<void()>(std::move(fn));
        g_idle_add_full(G_PRIORITY_DEFAULT, [](gpointer p)->gboolean {(*static_cast<std::function<void()>*>(p))();return G_SOURCE_REMOVE;}, task,
            [](gpointer p){delete static_cast<std::function<void()>*>(p);});
    }
};
static bool forceIteratorError=false;
static GstIteratorResult fixtureIteratorNext(GstIterator* iterator,GValue* value){if(forceIteratorError)return GST_ITERATOR_ERROR;return gst_iterator_next(iterator,value);}
#define gst_iterator_next fixtureIteratorNext
#define WEBKIT_DEFINE_ASYNC_DATA_STRUCT(T) static T* create##T(){return new T();} static void destroy##T(T* p){delete p;}
'''
CLASS = r'''
class MediaPlayerPrivateGStreamer : public std::enable_shared_from_this<MediaPlayerPrivateGStreamer> {
public:
    GRefPtr<GstElement> m_pipeline, m_downloadBuffer;
    PlayerPtr m_player; URL m_url;
    bool m_hasWebKitWebSrcSentEOS=false,m_isEndReached=false,m_isSegmentSeekAllowed=true,m_isChangingRate=false;
    bool m_isSeeking=false,m_isSeekPending=false,failed=false,playing=true,buffering=false,m_didErrorOccur=false,m_shouldResetPipeline=false;
    std::optional<bool> m_isLiveStream; SeekTarget m_seekTarget{{0}};
    enum class ChangePipelineStateResult { Failed, Succeeded };
    bool isMediaStreamPlayer(){return false;} MediaTime maxTimeSeekable(){return duration();}
    bool m_canFallBackToLastFinishedSeekPosition=false;
    void invalidateCachedPosition(){ } void timeChanged(MediaTime){ }
    bool m_shouldPreservePitch=true,m_isPipelinePlaying=true;
    float m_lastPlaybackRate=1,m_playbackRate=1;
    bool isPipelineWaitingPreroll(){return false;} MediaTime playbackPosition(){return currentTime();}
    int bufferingPercentage=100; double stateMilliseconds=0; MediaTime m_timeOfOverlappingSeek=MediaTime::invalidTime();
    explicit MediaPlayerPrivateGStreamer(GstElement* p):m_pipeline(p) { }
    GstElement* pipeline(){return m_pipeline.get();}
    bool paused(){return false;} bool isSeamlessSeekingEnabled(){return false;}
    MediaTime duration(){return {120};} MediaTime currentTime(){return m_isSeeking?m_seekTarget.time:MediaTime{0};} void didEnd(){ }
    void updateBufferingStatus(GstBufferingMode,double percent,bool,bool){bufferingPercentage=percent;buffering=percent<100;}
    ChangePipelineStateResult changePipelineState(GstState state){auto start=g_get_monotonic_time();gst_element_set_state(pipeline(),state);stateMilliseconds+=(g_get_monotonic_time()-start)/1000.0;playing=state==GST_STATE_PLAYING;return ChangePipelineStateResult::Succeeded;}
    void loadingFailed(MediaPlayer::NetworkState,MediaPlayer::ReadyState = MediaPlayer::ReadyState::HaveNothing,bool = true){failed=true;}
    bool doSeek(const SeekTarget&,float,bool=false,bool=false);
    void seekToTarget(const SeekTarget&); void finishSeek(); void updateStates(); void updatePlaybackRate();
};
ThreadSafeWeakPtr::ThreadSafeWeakPtr(MediaPlayerPrivateGStreamer& p):weak(p.weak_from_this()) { }
'''
INSTRUMENT = r'''
static gboolean fixtureSendEvent(GstElement*,GstEvent*);
#define gst_element_send_event fixtureSendEvent
#undef GST_ERROR_OBJECT
#define GST_ERROR_OBJECT(...) ((void)0)
#define g_object_set(...) ((void)0)
'''
TEST = r'''
#undef gst_element_send_event
struct Run {
    std::shared_ptr<MediaPlayerPrivateGStreamer> player;
    std::atomic<int> seeks{0}, activeSends{0}, maxActiveSends{0};
    std::atomic<GstClockTime> lastTarget{0}; std::atomic<double> lastRate{0};
    bool reject=false, replace=false, supersede=false, noop=false, asynchronous=true, duringRate=false, initialRate=false;
    int beats=0; gint64 previous=0,maxGap=0; GMainLoop* loop;
    GstPadEventFunction original=nullptr;
};
static gboolean fixtureSendEvent(GstElement* pipeline,GstEvent* event) {
    auto* r=static_cast<Run*>(g_object_get_data(G_OBJECT(pipeline),"fixture"));
    assert(r);int active=++r->activeSends;r->maxActiveSends.store(std::max(active,r->maxActiveSends.load()));
    gint64 target;gdouble rate;gst_event_parse_seek(event,&rate,nullptr,nullptr,nullptr,&target,nullptr,nullptr);r->lastTarget=target;r->lastRate=rate;
    g_usleep(40000); // Delay before FLUSH, so early PAUSED notification is exercised.
    bool result=gst_element_send_event(pipeline,event);--r->activeSends;return result;
}
static gboolean seekEvent(GstPad* pad,GstObject* parent,GstEvent* event) {
    auto* r=static_cast<Run*>(g_object_get_data(G_OBJECT(pad),"fixture"));
    bool isSeek=GST_EVENT_TYPE(event)==GST_EVENT_SEEK;
    if(isSeek) {r->seeks++;g_usleep(220000);if(r->reject && r->seeks==1){gst_event_unref(event);return false;}}
    bool result=r->original(pad,parent,event);
    if(result&&isSeek)g_idle_add([](gpointer data)->gboolean {auto* r=static_cast<Run*>(data);if(!r->replace)r->player->updateStates();return G_SOURCE_REMOVE;},r);
    return result;
}
static gboolean tick(gpointer data){auto* r=static_cast<Run*>(data);auto now=g_get_monotonic_time();if(r->previous)r->maxGap=std::max(r->maxGap,now-r->previous);r->previous=now;r->beats++;if(r->seeks||g_object_get_data(G_OBJECT(r->player->pipeline()),"localbooru-http-seek-dispatch-pending"))r->player->updateStates();return G_SOURCE_CONTINUE;}
static gboolean startSeek(gpointer data){auto* r=static_cast<Run*>(data);auto start=g_get_monotonic_time();r->player->m_isChangingRate=r->initialRate;r->player->seekToTarget({{2}});r->player->updateStates();if(r->asynchronous)assert(r->player->m_isSeeking);std::cout<<"dispatch_ms="<<(g_get_monotonic_time()-start)/1000.0<<" pause_ms="<<r->player->stateMilliseconds<<" ";return G_SOURCE_REMOVE;}
static gboolean changeOwner(gpointer data){auto* r=static_cast<Run*>(data);if(r->supersede){r->player->seekToTarget({{3}});r->player->seekToTarget({{3}});}if(r->noop)r->player->seekToTarget({{2}});if(r->duringRate){r->player->m_isChangingRate=true;r->player->m_playbackRate=2;r->player->updatePlaybackRate();assert(r->player->m_lastPlaybackRate==1);assert(r->player->m_isChangingRate);}
    if(r->replace)r->player->m_pipeline=adoptGRef(gst_pipeline_new("replacement"));return G_SOURCE_REMOVE;}
static gboolean end(gpointer data){g_main_loop_quit(static_cast<Run*>(data)->loop);return G_SOURCE_REMOVE;}
static void run(bool http,bool loop,bool rate,bool reject=false,bool supersede=false,bool replace=false,bool noop=false,bool duringRate=false,bool hls=false){
    Run r; r.reject=reject;r.supersede=supersede;r.replace=replace;r.noop=noop;r.duringRate=duringRate;r.initialRate=rate;r.asynchronous=http&&!loop&&!rate&&!hls;
    auto* pipeline=gst_parse_launch("videotestsrc name=source ! fakesink sync=true",nullptr);assert(pipeline);gst_object_ref(pipeline);
    g_object_set_data(G_OBJECT(pipeline),"fixture",&r);
    r.player=std::make_shared<MediaPlayerPrivateGStreamer>(pipeline);r.player->m_url.http=http;r.player->m_url.hls=hls;r.player->m_player.value.looping=loop;r.player->m_isChangingRate=rate;
    auto* source=gst_bin_get_by_name(GST_BIN(pipeline),"source");auto* pad=gst_element_get_static_pad(source,"src");r.original=GST_PAD_EVENTFUNC(pad);g_object_set_data(G_OBJECT(pad),"fixture",&r);gst_pad_set_event_function(pad,seekEvent);
    gst_element_set_state(pipeline,GST_STATE_PLAYING);gst_element_get_state(pipeline,nullptr,nullptr,5*GST_SECOND);
    r.loop=g_main_loop_new(nullptr,false);auto heartbeat=g_timeout_add(10,tick,&r);g_timeout_add(100,startSeek,&r);g_timeout_add(150,changeOwner,&r);g_timeout_add(800,end,&r);g_main_loop_run(r.loop);g_source_remove(heartbeat);
    std::cout<<"http="<<http<<" loop="<<loop<<" rate="<<rate<<" rejected="<<reject<<" superseded="<<supersede<<" replaced="<<replace<<" gap_ms="<<r.maxGap/1000.0<<" final_rate="<<r.lastRate<<" max_active="<<r.maxActiveSends<<" final_target_s="<<r.lastTarget/GST_SECOND<<" seeks="<<r.seeks<<" beats="<<r.beats<<"\n";
    assert(r.seeks==(supersede||duringRate?2:1));assert(r.maxActiveSends==1);
    assert(r.lastTarget==(supersede?3*GST_SECOND:2*GST_SECOND));if(r.asynchronous&&!duringRate){assert(r.maxGap<100000);assert(r.player->stateMilliseconds<100);}else assert(r.maxGap>200000);
    if(duringRate){assert(r.lastRate==2);assert(r.player->m_lastPlaybackRate==2);assert(!r.player->m_isChangingRate);}
    if(reject&&!supersede&&!replace){assert(r.player->failed);assert(!r.player->m_isSeeking);assert(!r.player->m_isSeekPending);assert(r.player->buffering);assert(!r.player->playing); /* terminal visible error, never pretend old buffer survived */}
    if(supersede||replace){assert(!r.player->failed);if(replace){assert(r.player->m_isSeeking);assert(!r.player->m_isSeekPending);}}
    gst_element_set_state(pipeline,GST_STATE_NULL);gst_object_unref(pad);gst_object_unref(source);gst_object_unref(pipeline);g_main_loop_unref(r.loop);
}
typedef struct { GstElement parent; } SyntheticInterpolation;
typedef struct { GstElementClass parent; } SyntheticInterpolationClass;
G_DEFINE_TYPE(SyntheticInterpolation,synthetic_interpolation,GST_TYPE_ELEMENT)
static void synthetic_interpolation_class_init(SyntheticInterpolationClass* klass){gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),"Synthetic interpolation","Filter/Video","Synthetic fixture only","DMC tests");}
static void synthetic_interpolation_init(SyntheticInterpolation*) { }
int main(int argc,char** argv){std::cout<<std::unitbuf;gst_init(&argc,&argv);
    auto* pipeline=gst_pipeline_new("guard");assert(localBooruShouldDispatchSeekAsync(pipeline,true,false,false));
    forceIteratorError=true;assert(!localBooruShouldDispatchSeekAsync(pipeline,true,false,false));forceIteratorError=false;
    assert(!localBooruShouldDispatchSeekAsync(pipeline,false,false,false));assert(!localBooruShouldDispatchSeekAsync(pipeline,true,true,false));assert(!localBooruShouldDispatchSeekAsync(pipeline,true,false,true));
    assert(gst_element_register(nullptr,"localbooruvs",GST_RANK_NONE,synthetic_interpolation_get_type()));
    auto* nested=gst_bin_new("nested");auto* filter=gst_element_factory_make("localbooruvs",nullptr);assert(filter);gst_bin_add(GST_BIN(nested),filter);gst_bin_add(GST_BIN(pipeline),nested);assert(!localBooruShouldDispatchSeekAsync(pipeline,true,false,false));gst_object_unref(pipeline);
    run(false,false,false);run(true,false,false);run(true,true,false);run(true,false,true);
    run(true,false,false,false,true);run(true,false,false,true);run(true,false,false,true,true);run(true,false,false,true,false,true);run(true,false,false,true,false,false,true);run(true,false,false,false,false,false,false,true);run(true,false,false,false,false,false,false,false,true);
    std::cout<<"Production HTTP guard/dispatch/pause/error/pending-ticket/owner tests passed\n";
}
'''

# AC: @responsive-original-stream-seeking ac-responsive-controls, ac-owner-boundary, ac-rejected-seek, ac-stream-routing
def main(source):
    media = (source / RELATIVE).read_text()
    helpers = media.split('// HTTP range seeks can wait', 1)[1].split('bool MediaPlayerPrivateGStreamer::doSeek', 1)[0]
    helpers = '// HTTP range seeks can wait' + helpers
    do_seek = 'bool MediaPlayerPrivateGStreamer::doSeek' + media.split('bool MediaPlayerPrivateGStreamer::doSeek', 1)[1].split('void MediaPlayerPrivateGStreamer::seekToTarget', 1)[0]
    seek_to = 'void MediaPlayerPrivateGStreamer::seekToTarget' + media.split('void MediaPlayerPrivateGStreamer::seekToTarget', 1)[1].split('void MediaPlayerPrivateGStreamer::updatePlaybackRate', 1)[0]
    state_tail = 'if (getStateResult == GST_STATE_CHANGE_SUCCESS && m_currentState >= GST_STATE_PAUSED)' + media.split('if (getStateResult == GST_STATE_CHANGE_SUCCESS && m_currentState >= GST_STATE_PAUSED)', 1)[1].split('void MediaPlayerPrivateGStreamer::mediaLocationChanged', 1)[0]
    update = 'void MediaPlayerPrivateGStreamer::updateStates(){ GstState state,pending; auto getStateResult=gst_element_get_state(m_pipeline.get(),&state,&pending,0); auto m_currentState=state; RefPtr player=m_player.get(); ' + state_tail
    finish = 'void MediaPlayerPrivateGStreamer::finishSeek' + media.split('void MediaPlayerPrivateGStreamer::finishSeek', 1)[1].split('void MediaPlayerPrivateGStreamer::updateStates', 1)[0]
    rate = 'void MediaPlayerPrivateGStreamer::updatePlaybackRate' + media.split('void MediaPlayerPrivateGStreamer::updatePlaybackRate', 1)[1].split('MediaTime MediaPlayerPrivateGStreamer::duration', 1)[0]
    with tempfile.TemporaryDirectory(prefix='dmc-synthetic-http-seek-') as directory:
        cpp = Path(directory) / 'fixture.cpp'
        cpp.write_text(HEADER + helpers + CLASS + INSTRUMENT + do_seek + seek_to + finish + rate + update + TEST)
        flags = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'gstreamer-1.0'], text=True))
        helper = os.environ.get('HOST_HEAVY_BUILD_HELPER', str(Path.home() / '.local/bin/host-heavy-build'))
        binary = Path(directory) / 'fixture'
        subprocess.run([helper, 'run', '--project', 'dmc-http-seek-fixture', '--worktree', str(REPO), '--wait', '21600', '--', 'g++', '-std=c++17', '-pthread', str(cpp), '-o', str(binary), *flags], check=True)
        subprocess.run([str(binary)], check=True, timeout=15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    main(parser.parse_args().source)
