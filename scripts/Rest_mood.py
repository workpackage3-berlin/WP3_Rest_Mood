#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Fri May 17 16:18:53 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from pylsl import StreamInfo, StreamOutlet
# Set up LabStreamingLayer stream.
info = StreamInfo(name='Psychopy', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='psy_marker')
outlet = StreamOutlet(info)  # Broadcast the stream.

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'Rest_mood'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['session'], expName, expInfo['date'])

    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/charlotte/Dropbox/Charite_PhD/tasks/Rest_Mood/Rest_mood.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Rest_Intro" ---
    Start_ButtonPress = keyboard.Keyboard()
    Rest_Intro_text = visual.TextStim(win=win, name='Rest_Intro_text',
        text='Bitte warten Sie, bis der Versuchsleiter die Ruhemessung startet.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Rest_Fixation" ---
    Rest_fix_cross = visual.ShapeStim(
        win=win, name='Rest_fix_cross', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Mood_Intro" ---
    Mood_intro_text = visual.TextStim(win=win, name='Mood_intro_text',
        text='Es folgen nun 4 Fragen zu Ihrem momentanen Gemütszustand.\n \nBitte beantworten Sie die Fragen verbal.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Ende_ruhemessung_text = visual.TextStim(win=win, name='Ende_ruhemessung_text',
        text='Ende der Ruhemessung\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Mood_Scale" ---
    Q_scale_text = visual.TextStim(win=win, name='Q_scale_text',
        text='Auf einer Skala von 1 bis 10, wie würden Sie Ihre Stimmung heute bewerten?\n\n\n\n1     2     3     4      5     6     7     8     9     10\n\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Q_scale_ButtonPress = keyboard.Keyboard()
    Q_unglucklich_text = visual.TextStim(win=win, name='Q_unglucklich_text',
        text='Sehr unglücklich',
        font='Open Sans',
        pos=(-0.5, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Q_glucklich_text = visual.TextStim(win=win, name='Q_glucklich_text',
        text='Sehr glücklich',
        font='Open Sans',
        pos=(0.5, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Mood_Describe" ---
    Q_describe_text = visual.TextStim(win=win, name='Q_describe_text',
        text='Wenn Sie an heute denken, wie würden Sie Ihre Stimmung beschreiben (z.B. glücklich, traurig, gestresst)? \n\nWelche Gefühle beschreiben Ihren aktuellen Zustand am Besten?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Q_describe_ButtonPress = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Mood_Influenced" ---
    Q_influenced_text = visual.TextStim(win=win, name='Q_influenced_text',
        text='Hat irgendetwas Bestimmtes Ihre Stimmung heute beeinflusst? \n\nWenn ja, bitte beschreiben.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Q_influenced_ButtonPress = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Mood_TasksToday" ---
    Q_taskToday_text = visual.TextStim(win=win, name='Q_taskToday_text',
        text='Was haben Sie heute gemacht/heute noch vor? ',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Q_taskToday_ButtonPress = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Mood_End" ---
    Mood_end_text = visual.TextStim(win=win, name='Mood_end_text',
        text='Ende der Fragen.\n\nBitte geben Sie der Versuchsleiterin Bescheid.\n\nVielen Dank. ',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Mood_end_buttonPress = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Rest_Intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest_Intro.started', globalClock.getTime())
    Start_ButtonPress.keys = []
    Start_ButtonPress.rt = []
    _Start_ButtonPress_allKeys = []
    # keep track of which components have finished
    Rest_IntroComponents = [Start_ButtonPress, Rest_Intro_text]
    for thisComponent in Rest_IntroComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest_Intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Start_ButtonPress* updates
        waitOnFlip = False
        
        # if Start_ButtonPress is starting this frame...
        if Start_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Start_ButtonPress.frameNStart = frameN  # exact frame index
            Start_ButtonPress.tStart = t  # local t and not account for scr refresh
            Start_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Start_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Start_ButtonPress.started')
            # update status
            Start_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Start_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Start_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Start_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = Start_ButtonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Start_ButtonPress_allKeys.extend(theseKeys)
            if len(_Start_ButtonPress_allKeys):
                Start_ButtonPress.keys = _Start_ButtonPress_allKeys[0].name  # just the first key pressed
                Start_ButtonPress.rt = _Start_ButtonPress_allKeys[0].rt
                Start_ButtonPress.duration = _Start_ButtonPress_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # *Rest_Intro_text* updates
        
        # if Rest_Intro_text is starting this frame...
        if Rest_Intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Rest_Intro_text.frameNStart = frameN  # exact frame index
            Rest_Intro_text.tStart = t  # local t and not account for scr refresh
            Rest_Intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Rest_Intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Rest_Intro_text.started')
            # update status
            Rest_Intro_text.status = STARTED
            Rest_Intro_text.setAutoDraw(True)
        
        # if Rest_Intro_text is active this frame...
        if Rest_Intro_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Rest_IntroComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest_Intro" ---
    for thisComponent in Rest_IntroComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest_Intro.stopped', globalClock.getTime())
    # check responses
    if Start_ButtonPress.keys in ['', [], None]:  # No response was made
        Start_ButtonPress.keys = None
    thisExp.addData('Start_ButtonPress.keys',Start_ButtonPress.keys)
    if Start_ButtonPress.keys != None:  # we had a response
        thisExp.addData('Start_ButtonPress.rt', Start_ButtonPress.rt)
        thisExp.addData('Start_ButtonPress.duration', Start_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "Rest_Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Rest_Fixation" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest_Fixation.started', globalClock.getTime())
    # keep track of which components have finished
    Rest_FixationComponents = [Rest_fix_cross]
    for thisComponent in Rest_FixationComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest_Fixation" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 310.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Rest_fix_cross* updates
        
        # if Rest_fix_cross is starting this frame...
        if Rest_fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:

            # Send LSL Marker : 
            mark = 'Rest_start'
            outlet.push_sample([mark])  # Push event marker.   

            # keep track of start time/frame for later
            Rest_fix_cross.frameNStart = frameN  # exact frame index
            Rest_fix_cross.tStart = t  # local t and not account for scr refresh
            Rest_fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Rest_fix_cross, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Rest_fix_cross.started')
            # update status
            Rest_fix_cross.status = STARTED
            Rest_fix_cross.setAutoDraw(True)
        
        # if Rest_fix_cross is active this frame...
        if Rest_fix_cross.status == STARTED:
            # update params
            pass
        
        # if Rest_fix_cross is stopping this frame...
        if Rest_fix_cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Rest_fix_cross.tStartRefresh + 310-frameTolerance:
                # keep track of stop time/frame for later
                Rest_fix_cross.tStop = t  # not accounting for scr refresh
                Rest_fix_cross.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest_fix_cross.stopped')
                # update status
                Rest_fix_cross.status = FINISHED
                Rest_fix_cross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Rest_FixationComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest_Fixation" ---
    for thisComponent in Rest_FixationComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest_Fixation.stopped', globalClock.getTime())

    # Send LSL Marker : 
    mark = 'Rest_end'
    outlet.push_sample([mark])  # Push event marker.   

    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-310.000000)
    
    # --- Prepare to start Routine "Mood_Intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_Intro.started', globalClock.getTime())
    # keep track of which components have finished
    Mood_IntroComponents = [Mood_intro_text, Ende_ruhemessung_text]
    for thisComponent in Mood_IntroComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_Intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Mood_intro_text* updates
        
        # if Mood_intro_text is starting this frame...
        if Mood_intro_text.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            Mood_intro_text.frameNStart = frameN  # exact frame index
            Mood_intro_text.tStart = t  # local t and not account for scr refresh
            Mood_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Mood_intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Mood_intro_text.started')
            # update status
            Mood_intro_text.status = STARTED
            Mood_intro_text.setAutoDraw(True)
        
        # if Mood_intro_text is active this frame...
        if Mood_intro_text.status == STARTED:
            # update params
            pass
        
        # if Mood_intro_text is stopping this frame...
        if Mood_intro_text.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 10-frameTolerance:
                # keep track of stop time/frame for later
                Mood_intro_text.tStop = t  # not accounting for scr refresh
                Mood_intro_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Mood_intro_text.stopped')
                # update status
                Mood_intro_text.status = FINISHED
                Mood_intro_text.setAutoDraw(False)
        
        # *Ende_ruhemessung_text* updates
        
        # if Ende_ruhemessung_text is starting this frame...
        if Ende_ruhemessung_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Ende_ruhemessung_text.frameNStart = frameN  # exact frame index
            Ende_ruhemessung_text.tStart = t  # local t and not account for scr refresh
            Ende_ruhemessung_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Ende_ruhemessung_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Ende_ruhemessung_text.started')
            # update status
            Ende_ruhemessung_text.status = STARTED
            Ende_ruhemessung_text.setAutoDraw(True)
        
        # if Ende_ruhemessung_text is active this frame...
        if Ende_ruhemessung_text.status == STARTED:
            # update params
            pass
        
        # if Ende_ruhemessung_text is stopping this frame...
        if Ende_ruhemessung_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Ende_ruhemessung_text.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Ende_ruhemessung_text.tStop = t  # not accounting for scr refresh
                Ende_ruhemessung_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Ende_ruhemessung_text.stopped')
                # update status
                Ende_ruhemessung_text.status = FINISHED
                Ende_ruhemessung_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_IntroComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_Intro" ---
    for thisComponent in Mood_IntroComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_Intro.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # --- Prepare to start Routine "Mood_Scale" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_Scale.started', globalClock.getTime())
    Q_scale_ButtonPress.keys = []
    Q_scale_ButtonPress.rt = []
    _Q_scale_ButtonPress_allKeys = []
    # keep track of which components have finished
    Mood_ScaleComponents = [Q_scale_text, Q_scale_ButtonPress, Q_unglucklich_text, Q_glucklich_text]
    for thisComponent in Mood_ScaleComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_Scale" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Q_scale_text* updates
        
        # if Q_scale_text is starting this frame...
        if Q_scale_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_scale_text.frameNStart = frameN  # exact frame index
            Q_scale_text.tStart = t  # local t and not account for scr refresh
            Q_scale_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_scale_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_scale_text.started')
            # update status
            Q_scale_text.status = STARTED
            Q_scale_text.setAutoDraw(True)
        
        # if Q_scale_text is active this frame...
        if Q_scale_text.status == STARTED:
            # update params
            pass
        
        # *Q_scale_ButtonPress* updates
        waitOnFlip = False
        
        # if Q_scale_ButtonPress is starting this frame...
        if Q_scale_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_scale_ButtonPress.frameNStart = frameN  # exact frame index
            Q_scale_ButtonPress.tStart = t  # local t and not account for scr refresh
            Q_scale_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_scale_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_scale_ButtonPress.started')
            # update status
            Q_scale_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Q_scale_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Q_scale_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if Q_scale_ButtonPress is stopping this frame...
        if Q_scale_ButtonPress.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Q_scale_ButtonPress.tStartRefresh + 90-frameTolerance:
                # keep track of stop time/frame for later
                Q_scale_ButtonPress.tStop = t  # not accounting for scr refresh
                Q_scale_ButtonPress.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Q_scale_ButtonPress.stopped')
                # update status
                Q_scale_ButtonPress.status = FINISHED
                Q_scale_ButtonPress.status = FINISHED
        if Q_scale_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = Q_scale_ButtonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Q_scale_ButtonPress_allKeys.extend(theseKeys)
            if len(_Q_scale_ButtonPress_allKeys):
                Q_scale_ButtonPress.keys = _Q_scale_ButtonPress_allKeys[0].name  # just the first key pressed
                Q_scale_ButtonPress.rt = _Q_scale_ButtonPress_allKeys[0].rt
                Q_scale_ButtonPress.duration = _Q_scale_ButtonPress_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # *Q_unglucklich_text* updates
        
        # if Q_unglucklich_text is starting this frame...
        if Q_unglucklich_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_unglucklich_text.frameNStart = frameN  # exact frame index
            Q_unglucklich_text.tStart = t  # local t and not account for scr refresh
            Q_unglucklich_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_unglucklich_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_unglucklich_text.started')
            # update status
            Q_unglucklich_text.status = STARTED
            Q_unglucklich_text.setAutoDraw(True)
        
        # if Q_unglucklich_text is active this frame...
        if Q_unglucklich_text.status == STARTED:
            # update params
            pass
        
        # *Q_glucklich_text* updates
        
        # if Q_glucklich_text is starting this frame...
        if Q_glucklich_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_glucklich_text.frameNStart = frameN  # exact frame index
            Q_glucklich_text.tStart = t  # local t and not account for scr refresh
            Q_glucklich_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_glucklich_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_glucklich_text.started')
            # update status
            Q_glucklich_text.status = STARTED
            Q_glucklich_text.setAutoDraw(True)
        
        # if Q_glucklich_text is active this frame...
        if Q_glucklich_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_ScaleComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_Scale" ---
    for thisComponent in Mood_ScaleComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_Scale.stopped', globalClock.getTime())
    # check responses
    if Q_scale_ButtonPress.keys in ['', [], None]:  # No response was made
        Q_scale_ButtonPress.keys = None
    thisExp.addData('Q_scale_ButtonPress.keys',Q_scale_ButtonPress.keys)
    if Q_scale_ButtonPress.keys != None:  # we had a response
        thisExp.addData('Q_scale_ButtonPress.rt', Q_scale_ButtonPress.rt)
        thisExp.addData('Q_scale_ButtonPress.duration', Q_scale_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "Mood_Scale" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Mood_Describe" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_Describe.started', globalClock.getTime())
    Q_describe_ButtonPress.keys = []
    Q_describe_ButtonPress.rt = []
    _Q_describe_ButtonPress_allKeys = []
    # keep track of which components have finished
    Mood_DescribeComponents = [Q_describe_text, Q_describe_ButtonPress]
    for thisComponent in Mood_DescribeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_Describe" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Q_describe_text* updates
        
        # if Q_describe_text is starting this frame...
        if Q_describe_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_describe_text.frameNStart = frameN  # exact frame index
            Q_describe_text.tStart = t  # local t and not account for scr refresh
            Q_describe_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_describe_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_describe_text.started')
            # update status
            Q_describe_text.status = STARTED
            Q_describe_text.setAutoDraw(True)
        
        # if Q_describe_text is active this frame...
        if Q_describe_text.status == STARTED:
            # update params
            pass
        
        # *Q_describe_ButtonPress* updates
        waitOnFlip = False
        
        # if Q_describe_ButtonPress is starting this frame...
        if Q_describe_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_describe_ButtonPress.frameNStart = frameN  # exact frame index
            Q_describe_ButtonPress.tStart = t  # local t and not account for scr refresh
            Q_describe_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_describe_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_describe_ButtonPress.started')
            # update status
            Q_describe_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Q_describe_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Q_describe_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if Q_describe_ButtonPress is stopping this frame...
        if Q_describe_ButtonPress.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Q_describe_ButtonPress.tStartRefresh + 90-frameTolerance:
                # keep track of stop time/frame for later
                Q_describe_ButtonPress.tStop = t  # not accounting for scr refresh
                Q_describe_ButtonPress.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Q_describe_ButtonPress.stopped')
                # update status
                Q_describe_ButtonPress.status = FINISHED
                Q_describe_ButtonPress.status = FINISHED
        if Q_describe_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = Q_describe_ButtonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Q_describe_ButtonPress_allKeys.extend(theseKeys)
            if len(_Q_describe_ButtonPress_allKeys):
                Q_describe_ButtonPress.keys = _Q_describe_ButtonPress_allKeys[-1].name  # just the last key pressed
                Q_describe_ButtonPress.rt = _Q_describe_ButtonPress_allKeys[-1].rt
                Q_describe_ButtonPress.duration = _Q_describe_ButtonPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_DescribeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_Describe" ---
    for thisComponent in Mood_DescribeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_Describe.stopped', globalClock.getTime())
    # check responses
    if Q_describe_ButtonPress.keys in ['', [], None]:  # No response was made
        Q_describe_ButtonPress.keys = None
    thisExp.addData('Q_describe_ButtonPress.keys',Q_describe_ButtonPress.keys)
    if Q_describe_ButtonPress.keys != None:  # we had a response
        thisExp.addData('Q_describe_ButtonPress.rt', Q_describe_ButtonPress.rt)
        thisExp.addData('Q_describe_ButtonPress.duration', Q_describe_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "Mood_Describe" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Mood_Influenced" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_Influenced.started', globalClock.getTime())
    Q_influenced_ButtonPress.keys = []
    Q_influenced_ButtonPress.rt = []
    _Q_influenced_ButtonPress_allKeys = []
    # keep track of which components have finished
    Mood_InfluencedComponents = [Q_influenced_text, Q_influenced_ButtonPress]
    for thisComponent in Mood_InfluencedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_Influenced" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Q_influenced_text* updates
        
        # if Q_influenced_text is starting this frame...
        if Q_influenced_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_influenced_text.frameNStart = frameN  # exact frame index
            Q_influenced_text.tStart = t  # local t and not account for scr refresh
            Q_influenced_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_influenced_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_influenced_text.started')
            # update status
            Q_influenced_text.status = STARTED
            Q_influenced_text.setAutoDraw(True)
        
        # if Q_influenced_text is active this frame...
        if Q_influenced_text.status == STARTED:
            # update params
            pass
        
        # *Q_influenced_ButtonPress* updates
        waitOnFlip = False
        
        # if Q_influenced_ButtonPress is starting this frame...
        if Q_influenced_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_influenced_ButtonPress.frameNStart = frameN  # exact frame index
            Q_influenced_ButtonPress.tStart = t  # local t and not account for scr refresh
            Q_influenced_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_influenced_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_influenced_ButtonPress.started')
            # update status
            Q_influenced_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Q_influenced_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Q_influenced_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if Q_influenced_ButtonPress is stopping this frame...
        if Q_influenced_ButtonPress.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Q_influenced_ButtonPress.tStartRefresh + 90-frameTolerance:
                # keep track of stop time/frame for later
                Q_influenced_ButtonPress.tStop = t  # not accounting for scr refresh
                Q_influenced_ButtonPress.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Q_influenced_ButtonPress.stopped')
                # update status
                Q_influenced_ButtonPress.status = FINISHED
                Q_influenced_ButtonPress.status = FINISHED
        if Q_influenced_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = Q_influenced_ButtonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Q_influenced_ButtonPress_allKeys.extend(theseKeys)
            if len(_Q_influenced_ButtonPress_allKeys):
                Q_influenced_ButtonPress.keys = _Q_influenced_ButtonPress_allKeys[0].name  # just the first key pressed
                Q_influenced_ButtonPress.rt = _Q_influenced_ButtonPress_allKeys[0].rt
                Q_influenced_ButtonPress.duration = _Q_influenced_ButtonPress_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_InfluencedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_Influenced" ---
    for thisComponent in Mood_InfluencedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_Influenced.stopped', globalClock.getTime())
    # check responses
    if Q_influenced_ButtonPress.keys in ['', [], None]:  # No response was made
        Q_influenced_ButtonPress.keys = None
    thisExp.addData('Q_influenced_ButtonPress.keys',Q_influenced_ButtonPress.keys)
    if Q_influenced_ButtonPress.keys != None:  # we had a response
        thisExp.addData('Q_influenced_ButtonPress.rt', Q_influenced_ButtonPress.rt)
        thisExp.addData('Q_influenced_ButtonPress.duration', Q_influenced_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "Mood_Influenced" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Mood_TasksToday" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_TasksToday.started', globalClock.getTime())
    Q_taskToday_ButtonPress.keys = []
    Q_taskToday_ButtonPress.rt = []
    _Q_taskToday_ButtonPress_allKeys = []
    # keep track of which components have finished
    Mood_TasksTodayComponents = [Q_taskToday_text, Q_taskToday_ButtonPress]
    for thisComponent in Mood_TasksTodayComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_TasksToday" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Q_taskToday_text* updates
        
        # if Q_taskToday_text is starting this frame...
        if Q_taskToday_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_taskToday_text.frameNStart = frameN  # exact frame index
            Q_taskToday_text.tStart = t  # local t and not account for scr refresh
            Q_taskToday_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_taskToday_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_taskToday_text.started')
            # update status
            Q_taskToday_text.status = STARTED
            Q_taskToday_text.setAutoDraw(True)
        
        # if Q_taskToday_text is active this frame...
        if Q_taskToday_text.status == STARTED:
            # update params
            pass
        
        # *Q_taskToday_ButtonPress* updates
        waitOnFlip = False
        
        # if Q_taskToday_ButtonPress is starting this frame...
        if Q_taskToday_ButtonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Q_taskToday_ButtonPress.frameNStart = frameN  # exact frame index
            Q_taskToday_ButtonPress.tStart = t  # local t and not account for scr refresh
            Q_taskToday_ButtonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Q_taskToday_ButtonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Q_taskToday_ButtonPress.started')
            # update status
            Q_taskToday_ButtonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Q_taskToday_ButtonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Q_taskToday_ButtonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if Q_taskToday_ButtonPress is stopping this frame...
        if Q_taskToday_ButtonPress.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Q_taskToday_ButtonPress.tStartRefresh + 90-frameTolerance:
                # keep track of stop time/frame for later
                Q_taskToday_ButtonPress.tStop = t  # not accounting for scr refresh
                Q_taskToday_ButtonPress.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Q_taskToday_ButtonPress.stopped')
                # update status
                Q_taskToday_ButtonPress.status = FINISHED
                Q_taskToday_ButtonPress.status = FINISHED
        if Q_taskToday_ButtonPress.status == STARTED and not waitOnFlip:
            theseKeys = Q_taskToday_ButtonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Q_taskToday_ButtonPress_allKeys.extend(theseKeys)
            if len(_Q_taskToday_ButtonPress_allKeys):
                Q_taskToday_ButtonPress.keys = _Q_taskToday_ButtonPress_allKeys[0].name  # just the first key pressed
                Q_taskToday_ButtonPress.rt = _Q_taskToday_ButtonPress_allKeys[0].rt
                Q_taskToday_ButtonPress.duration = _Q_taskToday_ButtonPress_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_TasksTodayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_TasksToday" ---
    for thisComponent in Mood_TasksTodayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_TasksToday.stopped', globalClock.getTime())
    # check responses
    if Q_taskToday_ButtonPress.keys in ['', [], None]:  # No response was made
        Q_taskToday_ButtonPress.keys = None
    thisExp.addData('Q_taskToday_ButtonPress.keys',Q_taskToday_ButtonPress.keys)
    if Q_taskToday_ButtonPress.keys != None:  # we had a response
        thisExp.addData('Q_taskToday_ButtonPress.rt', Q_taskToday_ButtonPress.rt)
        thisExp.addData('Q_taskToday_ButtonPress.duration', Q_taskToday_ButtonPress.duration)
    thisExp.nextEntry()
    # the Routine "Mood_TasksToday" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Mood_End" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Mood_End.started', globalClock.getTime())
    Mood_end_buttonPress.keys = []
    Mood_end_buttonPress.rt = []
    _Mood_end_buttonPress_allKeys = []
    # keep track of which components have finished
    Mood_EndComponents = [Mood_end_text, Mood_end_buttonPress]
    for thisComponent in Mood_EndComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Mood_End" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Mood_end_text* updates
        
        # if Mood_end_text is starting this frame...
        if Mood_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Mood_end_text.frameNStart = frameN  # exact frame index
            Mood_end_text.tStart = t  # local t and not account for scr refresh
            Mood_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Mood_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Mood_end_text.started')
            # update status
            Mood_end_text.status = STARTED
            Mood_end_text.setAutoDraw(True)
        
        # if Mood_end_text is active this frame...
        if Mood_end_text.status == STARTED:
            # update params
            pass
        
        # *Mood_end_buttonPress* updates
        waitOnFlip = False
        
        # if Mood_end_buttonPress is starting this frame...
        if Mood_end_buttonPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Mood_end_buttonPress.frameNStart = frameN  # exact frame index
            Mood_end_buttonPress.tStart = t  # local t and not account for scr refresh
            Mood_end_buttonPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Mood_end_buttonPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Mood_end_buttonPress.started')
            # update status
            Mood_end_buttonPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Mood_end_buttonPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Mood_end_buttonPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Mood_end_buttonPress.status == STARTED and not waitOnFlip:
            theseKeys = Mood_end_buttonPress.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _Mood_end_buttonPress_allKeys.extend(theseKeys)
            if len(_Mood_end_buttonPress_allKeys):
                Mood_end_buttonPress.keys = _Mood_end_buttonPress_allKeys[-1].name  # just the last key pressed
                Mood_end_buttonPress.rt = _Mood_end_buttonPress_allKeys[-1].rt
                Mood_end_buttonPress.duration = _Mood_end_buttonPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Mood_EndComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Mood_End" ---
    for thisComponent in Mood_EndComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Mood_End.stopped', globalClock.getTime())
    # check responses
    if Mood_end_buttonPress.keys in ['', [], None]:  # No response was made
        Mood_end_buttonPress.keys = None
    thisExp.addData('Mood_end_buttonPress.keys',Mood_end_buttonPress.keys)
    if Mood_end_buttonPress.keys != None:  # we had a response
        thisExp.addData('Mood_end_buttonPress.rt', Mood_end_buttonPress.rt)
        thisExp.addData('Mood_end_buttonPress.duration', Mood_end_buttonPress.duration)
    thisExp.nextEntry()
    # the Routine "Mood_End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
