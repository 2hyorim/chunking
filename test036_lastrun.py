#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 12월 13, 2024, at 23:44
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'test036'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
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
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\ailee\\OneDrive\\바탕 화면\\24-2\\실험심\\팀플\\프로그래밍 시도\\test\\test036_lastrun.py',
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
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
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
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_introduce') is None:
        # initialise key_resp_introduce
        key_resp_introduce = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_introduce',
        )
    if deviceManager.getDevice('key_resp_memory_test_0') is None:
        # initialise key_resp_memory_test_0
        key_resp_memory_test_0 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_memory_test_0',
        )
    # create speaker 'sound_1'
    deviceManager.addDevice(
        deviceName='sound_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_ready') is None:
        # initialise key_resp_ready
        key_resp_ready = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_ready',
        )
    if deviceManager.getDevice('key_resp_memory_test') is None:
        # initialise key_resp_memory_test
        key_resp_memory_test = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_memory_test',
        )
    # create speaker 'sound_2'
    deviceManager.addDevice(
        deviceName='sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_rest') is None:
        # initialise key_resp_rest
        key_resp_rest = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_rest',
        )
    if deviceManager.getDevice('key_resp_memory_test_2') is None:
        # initialise key_resp_memory_test_2
        key_resp_memory_test_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_memory_test_2',
        )
    # create speaker 'sound_3'
    deviceManager.addDevice(
        deviceName='sound_3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_memory_test_3') is None:
        # initialise key_resp_memory_test_3
        key_resp_memory_test_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_memory_test_3',
        )
    # create speaker 'sound_4'
    deviceManager.addDevice(
        deviceName='sound_4',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
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
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
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
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
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
    
    # --- Initialize components for Routine "intro" ---
    image_introduce = visual.ImageStim(
        win=win,
        name='image_introduce', 
        image='image_introduce.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_introduce = keyboard.Keyboard(deviceName='key_resp_introduce')
    
    # --- Initialize components for Routine "practice" ---
    focus_0 = visual.TextStim(win=win, name='focus_0',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_0 = visual.ImageStim(
        win=win,
        name='image_0', 
        image='image_practice.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.8, 0.8),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    blank_0 = visual.TextStim(win=win, name='blank_0',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_memorytest = visual.ImageStim(
        win=win,
        name='image_memorytest', 
        image='image_memorytest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.9, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_memory_test_0 = keyboard.Keyboard(deviceName='key_resp_memory_test_0')
    sound_1 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    
    # --- Initialize components for Routine "ready" ---
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', 
        image='image_ready.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.9, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_ready = keyboard.Keyboard(deviceName='key_resp_ready')
    
    # --- Initialize components for Routine "alphabet_random" ---
    focus = visual.TextStim(win=win, name='focus',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='image_random.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    blank = visual.TextStim(win=win, name='blank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_memorytest_2 = visual.ImageStim(
        win=win,
        name='image_memorytest_2', 
        image='image_memorytest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.9, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_memory_test = keyboard.Keyboard(deviceName='key_resp_memory_test')
    sound_2 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "rest" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', 
        image='image_rest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_rest = keyboard.Keyboard(deviceName='key_resp_rest')
    
    # --- Initialize components for Routine "alphabet_3" ---
    focus_2 = visual.TextStim(win=win, name='focus_2',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='image_3.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    blank_2 = visual.TextStim(win=win, name='blank_2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_memorytest_3 = visual.ImageStim(
        win=win,
        name='image_memorytest_3', 
        image='image_memorytest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.9, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_memory_test_2 = keyboard.Keyboard(deviceName='key_resp_memory_test_2')
    sound_3 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='sound_3',    name='sound_3'
    )
    sound_3.setVolume(1.0)
    
    # --- Initialize components for Routine "rest" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', 
        image='image_rest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_rest = keyboard.Keyboard(deviceName='key_resp_rest')
    
    # --- Initialize components for Routine "alphabet_6" ---
    focus_3 = visual.TextStim(win=win, name='focus_3',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='image_6.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    blank_3 = visual.TextStim(win=win, name='blank_3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_memorytest_4 = visual.ImageStim(
        win=win,
        name='image_memorytest_4', 
        image='image_memorytest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.9, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_memory_test_3 = keyboard.Keyboard(deviceName='key_resp_memory_test_3')
    sound_4 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='sound_4',    name='sound_4'
    )
    sound_4.setVolume(1.0)
    
    # --- Initialize components for Routine "finish" ---
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', 
        image='image_finish.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[image_introduce, key_resp_introduce],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_introduce
    key_resp_introduce.keys = []
    key_resp_introduce.rt = []
    _key_resp_introduce_allKeys = []
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
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
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_introduce* updates
        
        # if image_introduce is starting this frame...
        if image_introduce.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_introduce.frameNStart = frameN  # exact frame index
            image_introduce.tStart = t  # local t and not account for scr refresh
            image_introduce.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_introduce, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_introduce.started')
            # update status
            image_introduce.status = STARTED
            image_introduce.setAutoDraw(True)
        
        # if image_introduce is active this frame...
        if image_introduce.status == STARTED:
            # update params
            pass
        
        # *key_resp_introduce* updates
        waitOnFlip = False
        
        # if key_resp_introduce is starting this frame...
        if key_resp_introduce.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_introduce.frameNStart = frameN  # exact frame index
            key_resp_introduce.tStart = t  # local t and not account for scr refresh
            key_resp_introduce.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_introduce, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_introduce.started')
            # update status
            key_resp_introduce.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_introduce.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_introduce.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_introduce.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_introduce.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_introduce_allKeys.extend(theseKeys)
            if len(_key_resp_introduce_allKeys):
                key_resp_introduce.keys = _key_resp_introduce_allKeys[-1].name  # just the last key pressed
                key_resp_introduce.rt = _key_resp_introduce_allKeys[-1].rt
                key_resp_introduce.duration = _key_resp_introduce_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # check responses
    if key_resp_introduce.keys in ['', [], None]:  # No response was made
        key_resp_introduce.keys = None
    thisExp.addData('key_resp_introduce.keys',key_resp_introduce.keys)
    if key_resp_introduce.keys != None:  # we had a response
        thisExp.addData('key_resp_introduce.rt', key_resp_introduce.rt)
        thisExp.addData('key_resp_introduce.duration', key_resp_introduce.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice" ---
    # create an object to store info about Routine practice
    practice = data.Routine(
        name='practice',
        components=[focus_0, image_0, blank_0, image_memorytest, key_resp_memory_test_0, sound_1],
    )
    practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_memory_test_0
    key_resp_memory_test_0.keys = []
    key_resp_memory_test_0.rt = []
    _key_resp_memory_test_0_allKeys = []
    sound_1.setSound('B', secs=1.0, hamming=True)
    sound_1.setVolume(1.0, log=False)
    sound_1.seek(0)
    # store start times for practice
    practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice.tStart = globalClock.getTime(format='float')
    practice.status = STARTED
    thisExp.addData('practice.started', practice.tStart)
    practice.maxDuration = None
    # keep track of which components have finished
    practiceComponents = practice.components
    for thisComponent in practice.components:
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
    
    # --- Run Routine "practice" ---
    practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 228.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *focus_0* updates
        
        # if focus_0 is starting this frame...
        if focus_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            focus_0.frameNStart = frameN  # exact frame index
            focus_0.tStart = t  # local t and not account for scr refresh
            focus_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(focus_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'focus_0.started')
            # update status
            focus_0.status = STARTED
            focus_0.setAutoDraw(True)
        
        # if focus_0 is active this frame...
        if focus_0.status == STARTED:
            # update params
            pass
        
        # if focus_0 is stopping this frame...
        if focus_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > focus_0.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                focus_0.tStop = t  # not accounting for scr refresh
                focus_0.tStopRefresh = tThisFlipGlobal  # on global time
                focus_0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'focus_0.stopped')
                # update status
                focus_0.status = FINISHED
                focus_0.setAutoDraw(False)
        
        # *image_0* updates
        
        # if image_0 is starting this frame...
        if image_0.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            image_0.frameNStart = frameN  # exact frame index
            image_0.tStart = t  # local t and not account for scr refresh
            image_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_0.started')
            # update status
            image_0.status = STARTED
            image_0.setAutoDraw(True)
        
        # if image_0 is active this frame...
        if image_0.status == STARTED:
            # update params
            pass
        
        # if image_0 is stopping this frame...
        if image_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_0.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                image_0.tStop = t  # not accounting for scr refresh
                image_0.tStopRefresh = tThisFlipGlobal  # on global time
                image_0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_0.stopped')
                # update status
                image_0.status = FINISHED
                image_0.setAutoDraw(False)
        
        # *blank_0* updates
        
        # if blank_0 is starting this frame...
        if blank_0.status == NOT_STARTED and tThisFlip >= 46-frameTolerance:
            # keep track of start time/frame for later
            blank_0.frameNStart = frameN  # exact frame index
            blank_0.tStart = t  # local t and not account for scr refresh
            blank_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blank_0.started')
            # update status
            blank_0.status = STARTED
            blank_0.setAutoDraw(True)
        
        # if blank_0 is active this frame...
        if blank_0.status == STARTED:
            # update params
            pass
        
        # if blank_0 is stopping this frame...
        if blank_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blank_0.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                blank_0.tStop = t  # not accounting for scr refresh
                blank_0.tStopRefresh = tThisFlipGlobal  # on global time
                blank_0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_0.stopped')
                # update status
                blank_0.status = FINISHED
                blank_0.setAutoDraw(False)
        
        # *image_memorytest* updates
        
        # if image_memorytest is starting this frame...
        if image_memorytest.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            image_memorytest.frameNStart = frameN  # exact frame index
            image_memorytest.tStart = t  # local t and not account for scr refresh
            image_memorytest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_memorytest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_memorytest.started')
            # update status
            image_memorytest.status = STARTED
            image_memorytest.setAutoDraw(True)
        
        # if image_memorytest is active this frame...
        if image_memorytest.status == STARTED:
            # update params
            pass
        
        # if image_memorytest is stopping this frame...
        if image_memorytest.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_memorytest.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                image_memorytest.tStop = t  # not accounting for scr refresh
                image_memorytest.tStopRefresh = tThisFlipGlobal  # on global time
                image_memorytest.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_memorytest.stopped')
                # update status
                image_memorytest.status = FINISHED
                image_memorytest.setAutoDraw(False)
        
        # *key_resp_memory_test_0* updates
        waitOnFlip = False
        
        # if key_resp_memory_test_0 is starting this frame...
        if key_resp_memory_test_0.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            key_resp_memory_test_0.frameNStart = frameN  # exact frame index
            key_resp_memory_test_0.tStart = t  # local t and not account for scr refresh
            key_resp_memory_test_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_memory_test_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_memory_test_0.started')
            # update status
            key_resp_memory_test_0.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_memory_test_0.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_memory_test_0.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_memory_test_0 is stopping this frame...
        if key_resp_memory_test_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_memory_test_0.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_memory_test_0.tStop = t  # not accounting for scr refresh
                key_resp_memory_test_0.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_memory_test_0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_memory_test_0.stopped')
                # update status
                key_resp_memory_test_0.status = FINISHED
                key_resp_memory_test_0.status = FINISHED
        if key_resp_memory_test_0.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_memory_test_0.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_memory_test_0_allKeys.extend(theseKeys)
            if len(_key_resp_memory_test_0_allKeys):
                key_resp_memory_test_0.keys = _key_resp_memory_test_0_allKeys[-1].name  # just the last key pressed
                key_resp_memory_test_0.rt = _key_resp_memory_test_0_allKeys[-1].rt
                key_resp_memory_test_0.duration = _key_resp_memory_test_0_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *sound_1* updates
        
        # if sound_1 is starting this frame...
        if sound_1.status == NOT_STARTED and tThisFlip >= 227-frameTolerance:
            # keep track of start time/frame for later
            sound_1.frameNStart = frameN  # exact frame index
            sound_1.tStart = t  # local t and not account for scr refresh
            sound_1.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('sound_1.started', tThisFlipGlobal)
            # update status
            sound_1.status = STARTED
            sound_1.play(when=win)  # sync with win flip
        
        # if sound_1 is stopping this frame...
        if sound_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_1.tStartRefresh + 1.0-frameTolerance or sound_1.isFinished:
                # keep track of stop time/frame for later
                sound_1.tStop = t  # not accounting for scr refresh
                sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                sound_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sound_1.stopped')
                # update status
                sound_1.status = FINISHED
                sound_1.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[sound_1]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            practice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice" ---
    for thisComponent in practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice
    practice.tStop = globalClock.getTime(format='float')
    practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice.stopped', practice.tStop)
    # check responses
    if key_resp_memory_test_0.keys in ['', [], None]:  # No response was made
        key_resp_memory_test_0.keys = None
    thisExp.addData('key_resp_memory_test_0.keys',key_resp_memory_test_0.keys)
    if key_resp_memory_test_0.keys != None:  # we had a response
        thisExp.addData('key_resp_memory_test_0.rt', key_resp_memory_test_0.rt)
        thisExp.addData('key_resp_memory_test_0.duration', key_resp_memory_test_0.duration)
    sound_1.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if practice.maxDurationReached:
        routineTimer.addTime(-practice.maxDuration)
    elif practice.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-228.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "ready" ---
    # create an object to store info about Routine ready
    ready = data.Routine(
        name='ready',
        components=[image_4, key_resp_ready],
    )
    ready.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_ready
    key_resp_ready.keys = []
    key_resp_ready.rt = []
    _key_resp_ready_allKeys = []
    # store start times for ready
    ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ready.tStart = globalClock.getTime(format='float')
    ready.status = STARTED
    thisExp.addData('ready.started', ready.tStart)
    ready.maxDuration = None
    # keep track of which components have finished
    readyComponents = ready.components
    for thisComponent in ready.components:
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
    
    # --- Run Routine "ready" ---
    ready.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_4* updates
        
        # if image_4 is starting this frame...
        if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_4.frameNStart = frameN  # exact frame index
            image_4.tStart = t  # local t and not account for scr refresh
            image_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_4.started')
            # update status
            image_4.status = STARTED
            image_4.setAutoDraw(True)
        
        # if image_4 is active this frame...
        if image_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_ready* updates
        waitOnFlip = False
        
        # if key_resp_ready is starting this frame...
        if key_resp_ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ready.frameNStart = frameN  # exact frame index
            key_resp_ready.tStart = t  # local t and not account for scr refresh
            key_resp_ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ready.started')
            # update status
            key_resp_ready.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ready.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_ready.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ready.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ready_allKeys.extend(theseKeys)
            if len(_key_resp_ready_allKeys):
                key_resp_ready.keys = _key_resp_ready_allKeys[-1].name  # just the last key pressed
                key_resp_ready.rt = _key_resp_ready_allKeys[-1].rt
                key_resp_ready.duration = _key_resp_ready_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ready.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ready.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ready" ---
    for thisComponent in ready.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ready
    ready.tStop = globalClock.getTime(format='float')
    ready.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ready.stopped', ready.tStop)
    # check responses
    if key_resp_ready.keys in ['', [], None]:  # No response was made
        key_resp_ready.keys = None
    thisExp.addData('key_resp_ready.keys',key_resp_ready.keys)
    if key_resp_ready.keys != None:  # we had a response
        thisExp.addData('key_resp_ready.rt', key_resp_ready.rt)
        thisExp.addData('key_resp_ready.duration', key_resp_ready.duration)
    thisExp.nextEntry()
    # the Routine "ready" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "alphabet_random" ---
    # create an object to store info about Routine alphabet_random
    alphabet_random = data.Routine(
        name='alphabet_random',
        components=[focus, image, blank, image_memorytest_2, key_resp_memory_test, sound_2],
    )
    alphabet_random.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_memory_test
    key_resp_memory_test.keys = []
    key_resp_memory_test.rt = []
    _key_resp_memory_test_allKeys = []
    sound_2.setSound('B', secs=1.0, hamming=True)
    sound_2.setVolume(1.0, log=False)
    sound_2.seek(0)
    # store start times for alphabet_random
    alphabet_random.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    alphabet_random.tStart = globalClock.getTime(format='float')
    alphabet_random.status = STARTED
    thisExp.addData('alphabet_random.started', alphabet_random.tStart)
    alphabet_random.maxDuration = None
    # keep track of which components have finished
    alphabet_randomComponents = alphabet_random.components
    for thisComponent in alphabet_random.components:
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
    
    # --- Run Routine "alphabet_random" ---
    alphabet_random.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 228.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *focus* updates
        
        # if focus is starting this frame...
        if focus.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            focus.frameNStart = frameN  # exact frame index
            focus.tStart = t  # local t and not account for scr refresh
            focus.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(focus, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'focus.started')
            # update status
            focus.status = STARTED
            focus.setAutoDraw(True)
        
        # if focus is active this frame...
        if focus.status == STARTED:
            # update params
            pass
        
        # if focus is stopping this frame...
        if focus.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > focus.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                focus.tStop = t  # not accounting for scr refresh
                focus.tStopRefresh = tThisFlipGlobal  # on global time
                focus.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'focus.stopped')
                # update status
                focus.status = FINISHED
                focus.setAutoDraw(False)
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        
        # if image is stopping this frame...
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.tStopRefresh = tThisFlipGlobal  # on global time
                image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.stopped')
                # update status
                image.status = FINISHED
                image.setAutoDraw(False)
        
        # *blank* updates
        
        # if blank is starting this frame...
        if blank.status == NOT_STARTED and tThisFlip >= 46-frameTolerance:
            # keep track of start time/frame for later
            blank.frameNStart = frameN  # exact frame index
            blank.tStart = t  # local t and not account for scr refresh
            blank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blank.started')
            # update status
            blank.status = STARTED
            blank.setAutoDraw(True)
        
        # if blank is active this frame...
        if blank.status == STARTED:
            # update params
            pass
        
        # if blank is stopping this frame...
        if blank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blank.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                blank.tStop = t  # not accounting for scr refresh
                blank.tStopRefresh = tThisFlipGlobal  # on global time
                blank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank.stopped')
                # update status
                blank.status = FINISHED
                blank.setAutoDraw(False)
        
        # *image_memorytest_2* updates
        
        # if image_memorytest_2 is starting this frame...
        if image_memorytest_2.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            image_memorytest_2.frameNStart = frameN  # exact frame index
            image_memorytest_2.tStart = t  # local t and not account for scr refresh
            image_memorytest_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_memorytest_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_memorytest_2.started')
            # update status
            image_memorytest_2.status = STARTED
            image_memorytest_2.setAutoDraw(True)
        
        # if image_memorytest_2 is active this frame...
        if image_memorytest_2.status == STARTED:
            # update params
            pass
        
        # if image_memorytest_2 is stopping this frame...
        if image_memorytest_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_memorytest_2.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                image_memorytest_2.tStop = t  # not accounting for scr refresh
                image_memorytest_2.tStopRefresh = tThisFlipGlobal  # on global time
                image_memorytest_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_memorytest_2.stopped')
                # update status
                image_memorytest_2.status = FINISHED
                image_memorytest_2.setAutoDraw(False)
        
        # *key_resp_memory_test* updates
        waitOnFlip = False
        
        # if key_resp_memory_test is starting this frame...
        if key_resp_memory_test.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            key_resp_memory_test.frameNStart = frameN  # exact frame index
            key_resp_memory_test.tStart = t  # local t and not account for scr refresh
            key_resp_memory_test.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_memory_test, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_memory_test.started')
            # update status
            key_resp_memory_test.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_memory_test.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_memory_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_memory_test is stopping this frame...
        if key_resp_memory_test.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_memory_test.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_memory_test.tStop = t  # not accounting for scr refresh
                key_resp_memory_test.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_memory_test.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_memory_test.stopped')
                # update status
                key_resp_memory_test.status = FINISHED
                key_resp_memory_test.status = FINISHED
        if key_resp_memory_test.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_memory_test.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_memory_test_allKeys.extend(theseKeys)
            if len(_key_resp_memory_test_allKeys):
                key_resp_memory_test.keys = _key_resp_memory_test_allKeys[-1].name  # just the last key pressed
                key_resp_memory_test.rt = _key_resp_memory_test_allKeys[-1].rt
                key_resp_memory_test.duration = _key_resp_memory_test_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *sound_2* updates
        
        # if sound_2 is starting this frame...
        if sound_2.status == NOT_STARTED and tThisFlip >= 227-frameTolerance:
            # keep track of start time/frame for later
            sound_2.frameNStart = frameN  # exact frame index
            sound_2.tStart = t  # local t and not account for scr refresh
            sound_2.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('sound_2.started', tThisFlipGlobal)
            # update status
            sound_2.status = STARTED
            sound_2.play(when=win)  # sync with win flip
        
        # if sound_2 is stopping this frame...
        if sound_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_2.tStartRefresh + 1.0-frameTolerance or sound_2.isFinished:
                # keep track of stop time/frame for later
                sound_2.tStop = t  # not accounting for scr refresh
                sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                sound_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sound_2.stopped')
                # update status
                sound_2.status = FINISHED
                sound_2.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[sound_2]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            alphabet_random.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in alphabet_random.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "alphabet_random" ---
    for thisComponent in alphabet_random.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for alphabet_random
    alphabet_random.tStop = globalClock.getTime(format='float')
    alphabet_random.tStopRefresh = tThisFlipGlobal
    thisExp.addData('alphabet_random.stopped', alphabet_random.tStop)
    # check responses
    if key_resp_memory_test.keys in ['', [], None]:  # No response was made
        key_resp_memory_test.keys = None
    thisExp.addData('key_resp_memory_test.keys',key_resp_memory_test.keys)
    if key_resp_memory_test.keys != None:  # we had a response
        thisExp.addData('key_resp_memory_test.rt', key_resp_memory_test.rt)
        thisExp.addData('key_resp_memory_test.duration', key_resp_memory_test.duration)
    sound_2.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if alphabet_random.maxDurationReached:
        routineTimer.addTime(-alphabet_random.maxDuration)
    elif alphabet_random.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-228.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "rest" ---
    # create an object to store info about Routine rest
    rest = data.Routine(
        name='rest',
        components=[image_5, key_resp_rest],
    )
    rest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_rest
    key_resp_rest.keys = []
    key_resp_rest.rt = []
    _key_resp_rest_allKeys = []
    # store start times for rest
    rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    rest.tStart = globalClock.getTime(format='float')
    rest.status = STARTED
    thisExp.addData('rest.started', rest.tStart)
    rest.maxDuration = None
    # keep track of which components have finished
    restComponents = rest.components
    for thisComponent in rest.components:
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
    
    # --- Run Routine "rest" ---
    rest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_5* updates
        
        # if image_5 is starting this frame...
        if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_5.frameNStart = frameN  # exact frame index
            image_5.tStart = t  # local t and not account for scr refresh
            image_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_5.started')
            # update status
            image_5.status = STARTED
            image_5.setAutoDraw(True)
        
        # if image_5 is active this frame...
        if image_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_rest* updates
        waitOnFlip = False
        
        # if key_resp_rest is starting this frame...
        if key_resp_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_rest.frameNStart = frameN  # exact frame index
            key_resp_rest.tStart = t  # local t and not account for scr refresh
            key_resp_rest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_rest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_rest.started')
            # update status
            key_resp_rest.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_rest.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_rest.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_rest.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_rest.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_rest_allKeys.extend(theseKeys)
            if len(_key_resp_rest_allKeys):
                key_resp_rest.keys = _key_resp_rest_allKeys[-1].name  # just the last key pressed
                key_resp_rest.rt = _key_resp_rest_allKeys[-1].rt
                key_resp_rest.duration = _key_resp_rest_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            rest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in rest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in rest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for rest
    rest.tStop = globalClock.getTime(format='float')
    rest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('rest.stopped', rest.tStop)
    # check responses
    if key_resp_rest.keys in ['', [], None]:  # No response was made
        key_resp_rest.keys = None
    thisExp.addData('key_resp_rest.keys',key_resp_rest.keys)
    if key_resp_rest.keys != None:  # we had a response
        thisExp.addData('key_resp_rest.rt', key_resp_rest.rt)
        thisExp.addData('key_resp_rest.duration', key_resp_rest.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "alphabet_3" ---
    # create an object to store info about Routine alphabet_3
    alphabet_3 = data.Routine(
        name='alphabet_3',
        components=[focus_2, image_2, blank_2, image_memorytest_3, key_resp_memory_test_2, sound_3],
    )
    alphabet_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_memory_test_2
    key_resp_memory_test_2.keys = []
    key_resp_memory_test_2.rt = []
    _key_resp_memory_test_2_allKeys = []
    sound_3.setSound('B', secs=1.0, hamming=True)
    sound_3.setVolume(1.0, log=False)
    sound_3.seek(0)
    # store start times for alphabet_3
    alphabet_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    alphabet_3.tStart = globalClock.getTime(format='float')
    alphabet_3.status = STARTED
    thisExp.addData('alphabet_3.started', alphabet_3.tStart)
    alphabet_3.maxDuration = None
    # keep track of which components have finished
    alphabet_3Components = alphabet_3.components
    for thisComponent in alphabet_3.components:
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
    
    # --- Run Routine "alphabet_3" ---
    alphabet_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 228.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *focus_2* updates
        
        # if focus_2 is starting this frame...
        if focus_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            focus_2.frameNStart = frameN  # exact frame index
            focus_2.tStart = t  # local t and not account for scr refresh
            focus_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(focus_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'focus_2.started')
            # update status
            focus_2.status = STARTED
            focus_2.setAutoDraw(True)
        
        # if focus_2 is active this frame...
        if focus_2.status == STARTED:
            # update params
            pass
        
        # if focus_2 is stopping this frame...
        if focus_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > focus_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                focus_2.tStop = t  # not accounting for scr refresh
                focus_2.tStopRefresh = tThisFlipGlobal  # on global time
                focus_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'focus_2.stopped')
                # update status
                focus_2.status = FINISHED
                focus_2.setAutoDraw(False)
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            pass
        
        # if image_2 is stopping this frame...
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.tStopRefresh = tThisFlipGlobal  # on global time
                image_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.stopped')
                # update status
                image_2.status = FINISHED
                image_2.setAutoDraw(False)
        
        # *blank_2* updates
        
        # if blank_2 is starting this frame...
        if blank_2.status == NOT_STARTED and tThisFlip >= 46-frameTolerance:
            # keep track of start time/frame for later
            blank_2.frameNStart = frameN  # exact frame index
            blank_2.tStart = t  # local t and not account for scr refresh
            blank_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blank_2.started')
            # update status
            blank_2.status = STARTED
            blank_2.setAutoDraw(True)
        
        # if blank_2 is active this frame...
        if blank_2.status == STARTED:
            # update params
            pass
        
        # if blank_2 is stopping this frame...
        if blank_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blank_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                blank_2.tStop = t  # not accounting for scr refresh
                blank_2.tStopRefresh = tThisFlipGlobal  # on global time
                blank_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_2.stopped')
                # update status
                blank_2.status = FINISHED
                blank_2.setAutoDraw(False)
        
        # *image_memorytest_3* updates
        
        # if image_memorytest_3 is starting this frame...
        if image_memorytest_3.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            image_memorytest_3.frameNStart = frameN  # exact frame index
            image_memorytest_3.tStart = t  # local t and not account for scr refresh
            image_memorytest_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_memorytest_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_memorytest_3.started')
            # update status
            image_memorytest_3.status = STARTED
            image_memorytest_3.setAutoDraw(True)
        
        # if image_memorytest_3 is active this frame...
        if image_memorytest_3.status == STARTED:
            # update params
            pass
        
        # if image_memorytest_3 is stopping this frame...
        if image_memorytest_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_memorytest_3.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                image_memorytest_3.tStop = t  # not accounting for scr refresh
                image_memorytest_3.tStopRefresh = tThisFlipGlobal  # on global time
                image_memorytest_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_memorytest_3.stopped')
                # update status
                image_memorytest_3.status = FINISHED
                image_memorytest_3.setAutoDraw(False)
        
        # *key_resp_memory_test_2* updates
        waitOnFlip = False
        
        # if key_resp_memory_test_2 is starting this frame...
        if key_resp_memory_test_2.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            key_resp_memory_test_2.frameNStart = frameN  # exact frame index
            key_resp_memory_test_2.tStart = t  # local t and not account for scr refresh
            key_resp_memory_test_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_memory_test_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_memory_test_2.started')
            # update status
            key_resp_memory_test_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_memory_test_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_memory_test_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_memory_test_2 is stopping this frame...
        if key_resp_memory_test_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_memory_test_2.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_memory_test_2.tStop = t  # not accounting for scr refresh
                key_resp_memory_test_2.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_memory_test_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_memory_test_2.stopped')
                # update status
                key_resp_memory_test_2.status = FINISHED
                key_resp_memory_test_2.status = FINISHED
        if key_resp_memory_test_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_memory_test_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_memory_test_2_allKeys.extend(theseKeys)
            if len(_key_resp_memory_test_2_allKeys):
                key_resp_memory_test_2.keys = _key_resp_memory_test_2_allKeys[-1].name  # just the last key pressed
                key_resp_memory_test_2.rt = _key_resp_memory_test_2_allKeys[-1].rt
                key_resp_memory_test_2.duration = _key_resp_memory_test_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *sound_3* updates
        
        # if sound_3 is starting this frame...
        if sound_3.status == NOT_STARTED and tThisFlip >= 227-frameTolerance:
            # keep track of start time/frame for later
            sound_3.frameNStart = frameN  # exact frame index
            sound_3.tStart = t  # local t and not account for scr refresh
            sound_3.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('sound_3.started', tThisFlipGlobal)
            # update status
            sound_3.status = STARTED
            sound_3.play(when=win)  # sync with win flip
        
        # if sound_3 is stopping this frame...
        if sound_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_3.tStartRefresh + 1.0-frameTolerance or sound_3.isFinished:
                # keep track of stop time/frame for later
                sound_3.tStop = t  # not accounting for scr refresh
                sound_3.tStopRefresh = tThisFlipGlobal  # on global time
                sound_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sound_3.stopped')
                # update status
                sound_3.status = FINISHED
                sound_3.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[sound_3]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            alphabet_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in alphabet_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "alphabet_3" ---
    for thisComponent in alphabet_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for alphabet_3
    alphabet_3.tStop = globalClock.getTime(format='float')
    alphabet_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('alphabet_3.stopped', alphabet_3.tStop)
    # check responses
    if key_resp_memory_test_2.keys in ['', [], None]:  # No response was made
        key_resp_memory_test_2.keys = None
    thisExp.addData('key_resp_memory_test_2.keys',key_resp_memory_test_2.keys)
    if key_resp_memory_test_2.keys != None:  # we had a response
        thisExp.addData('key_resp_memory_test_2.rt', key_resp_memory_test_2.rt)
        thisExp.addData('key_resp_memory_test_2.duration', key_resp_memory_test_2.duration)
    sound_3.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if alphabet_3.maxDurationReached:
        routineTimer.addTime(-alphabet_3.maxDuration)
    elif alphabet_3.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-228.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "rest" ---
    # create an object to store info about Routine rest
    rest = data.Routine(
        name='rest',
        components=[image_5, key_resp_rest],
    )
    rest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_rest
    key_resp_rest.keys = []
    key_resp_rest.rt = []
    _key_resp_rest_allKeys = []
    # store start times for rest
    rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    rest.tStart = globalClock.getTime(format='float')
    rest.status = STARTED
    thisExp.addData('rest.started', rest.tStart)
    rest.maxDuration = None
    # keep track of which components have finished
    restComponents = rest.components
    for thisComponent in rest.components:
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
    
    # --- Run Routine "rest" ---
    rest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_5* updates
        
        # if image_5 is starting this frame...
        if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_5.frameNStart = frameN  # exact frame index
            image_5.tStart = t  # local t and not account for scr refresh
            image_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_5.started')
            # update status
            image_5.status = STARTED
            image_5.setAutoDraw(True)
        
        # if image_5 is active this frame...
        if image_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_rest* updates
        waitOnFlip = False
        
        # if key_resp_rest is starting this frame...
        if key_resp_rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_rest.frameNStart = frameN  # exact frame index
            key_resp_rest.tStart = t  # local t and not account for scr refresh
            key_resp_rest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_rest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_rest.started')
            # update status
            key_resp_rest.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_rest.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_rest.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_rest.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_rest.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_rest_allKeys.extend(theseKeys)
            if len(_key_resp_rest_allKeys):
                key_resp_rest.keys = _key_resp_rest_allKeys[-1].name  # just the last key pressed
                key_resp_rest.rt = _key_resp_rest_allKeys[-1].rt
                key_resp_rest.duration = _key_resp_rest_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            rest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in rest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in rest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for rest
    rest.tStop = globalClock.getTime(format='float')
    rest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('rest.stopped', rest.tStop)
    # check responses
    if key_resp_rest.keys in ['', [], None]:  # No response was made
        key_resp_rest.keys = None
    thisExp.addData('key_resp_rest.keys',key_resp_rest.keys)
    if key_resp_rest.keys != None:  # we had a response
        thisExp.addData('key_resp_rest.rt', key_resp_rest.rt)
        thisExp.addData('key_resp_rest.duration', key_resp_rest.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "alphabet_6" ---
    # create an object to store info about Routine alphabet_6
    alphabet_6 = data.Routine(
        name='alphabet_6',
        components=[focus_3, image_3, blank_3, image_memorytest_4, key_resp_memory_test_3, sound_4],
    )
    alphabet_6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_memory_test_3
    key_resp_memory_test_3.keys = []
    key_resp_memory_test_3.rt = []
    _key_resp_memory_test_3_allKeys = []
    sound_4.setSound('B', secs=1.0, hamming=True)
    sound_4.setVolume(1.0, log=False)
    sound_4.seek(0)
    # store start times for alphabet_6
    alphabet_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    alphabet_6.tStart = globalClock.getTime(format='float')
    alphabet_6.status = STARTED
    thisExp.addData('alphabet_6.started', alphabet_6.tStart)
    alphabet_6.maxDuration = None
    # keep track of which components have finished
    alphabet_6Components = alphabet_6.components
    for thisComponent in alphabet_6.components:
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
    
    # --- Run Routine "alphabet_6" ---
    alphabet_6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 228.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *focus_3* updates
        
        # if focus_3 is starting this frame...
        if focus_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            focus_3.frameNStart = frameN  # exact frame index
            focus_3.tStart = t  # local t and not account for scr refresh
            focus_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(focus_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'focus_3.started')
            # update status
            focus_3.status = STARTED
            focus_3.setAutoDraw(True)
        
        # if focus_3 is active this frame...
        if focus_3.status == STARTED:
            # update params
            pass
        
        # if focus_3 is stopping this frame...
        if focus_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > focus_3.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                focus_3.tStop = t  # not accounting for scr refresh
                focus_3.tStopRefresh = tThisFlipGlobal  # on global time
                focus_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'focus_3.stopped')
                # update status
                focus_3.status = FINISHED
                focus_3.setAutoDraw(False)
        
        # *image_3* updates
        
        # if image_3 is starting this frame...
        if image_3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            image_3.frameNStart = frameN  # exact frame index
            image_3.tStart = t  # local t and not account for scr refresh
            image_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_3.started')
            # update status
            image_3.status = STARTED
            image_3.setAutoDraw(True)
        
        # if image_3 is active this frame...
        if image_3.status == STARTED:
            # update params
            pass
        
        # if image_3 is stopping this frame...
        if image_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_3.tStartRefresh + 45-frameTolerance:
                # keep track of stop time/frame for later
                image_3.tStop = t  # not accounting for scr refresh
                image_3.tStopRefresh = tThisFlipGlobal  # on global time
                image_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_3.stopped')
                # update status
                image_3.status = FINISHED
                image_3.setAutoDraw(False)
        
        # *blank_3* updates
        
        # if blank_3 is starting this frame...
        if blank_3.status == NOT_STARTED and tThisFlip >= 46-frameTolerance:
            # keep track of start time/frame for later
            blank_3.frameNStart = frameN  # exact frame index
            blank_3.tStart = t  # local t and not account for scr refresh
            blank_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blank_3.started')
            # update status
            blank_3.status = STARTED
            blank_3.setAutoDraw(True)
        
        # if blank_3 is active this frame...
        if blank_3.status == STARTED:
            # update params
            pass
        
        # if blank_3 is stopping this frame...
        if blank_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blank_3.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                blank_3.tStop = t  # not accounting for scr refresh
                blank_3.tStopRefresh = tThisFlipGlobal  # on global time
                blank_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_3.stopped')
                # update status
                blank_3.status = FINISHED
                blank_3.setAutoDraw(False)
        
        # *image_memorytest_4* updates
        
        # if image_memorytest_4 is starting this frame...
        if image_memorytest_4.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            image_memorytest_4.frameNStart = frameN  # exact frame index
            image_memorytest_4.tStart = t  # local t and not account for scr refresh
            image_memorytest_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_memorytest_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_memorytest_4.started')
            # update status
            image_memorytest_4.status = STARTED
            image_memorytest_4.setAutoDraw(True)
        
        # if image_memorytest_4 is active this frame...
        if image_memorytest_4.status == STARTED:
            # update params
            pass
        
        # if image_memorytest_4 is stopping this frame...
        if image_memorytest_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_memorytest_4.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                image_memorytest_4.tStop = t  # not accounting for scr refresh
                image_memorytest_4.tStopRefresh = tThisFlipGlobal  # on global time
                image_memorytest_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_memorytest_4.stopped')
                # update status
                image_memorytest_4.status = FINISHED
                image_memorytest_4.setAutoDraw(False)
        
        # *key_resp_memory_test_3* updates
        waitOnFlip = False
        
        # if key_resp_memory_test_3 is starting this frame...
        if key_resp_memory_test_3.status == NOT_STARTED and tThisFlip >= 47-frameTolerance:
            # keep track of start time/frame for later
            key_resp_memory_test_3.frameNStart = frameN  # exact frame index
            key_resp_memory_test_3.tStart = t  # local t and not account for scr refresh
            key_resp_memory_test_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_memory_test_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_memory_test_3.started')
            # update status
            key_resp_memory_test_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_memory_test_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_memory_test_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_memory_test_3 is stopping this frame...
        if key_resp_memory_test_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_memory_test_3.tStartRefresh + 180-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_memory_test_3.tStop = t  # not accounting for scr refresh
                key_resp_memory_test_3.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_memory_test_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_memory_test_3.stopped')
                # update status
                key_resp_memory_test_3.status = FINISHED
                key_resp_memory_test_3.status = FINISHED
        if key_resp_memory_test_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_memory_test_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_memory_test_3_allKeys.extend(theseKeys)
            if len(_key_resp_memory_test_3_allKeys):
                key_resp_memory_test_3.keys = _key_resp_memory_test_3_allKeys[-1].name  # just the last key pressed
                key_resp_memory_test_3.rt = _key_resp_memory_test_3_allKeys[-1].rt
                key_resp_memory_test_3.duration = _key_resp_memory_test_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *sound_4* updates
        
        # if sound_4 is starting this frame...
        if sound_4.status == NOT_STARTED and tThisFlip >= 227-frameTolerance:
            # keep track of start time/frame for later
            sound_4.frameNStart = frameN  # exact frame index
            sound_4.tStart = t  # local t and not account for scr refresh
            sound_4.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('sound_4.started', tThisFlipGlobal)
            # update status
            sound_4.status = STARTED
            sound_4.play(when=win)  # sync with win flip
        
        # if sound_4 is stopping this frame...
        if sound_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_4.tStartRefresh + 1.0-frameTolerance or sound_4.isFinished:
                # keep track of stop time/frame for later
                sound_4.tStop = t  # not accounting for scr refresh
                sound_4.tStopRefresh = tThisFlipGlobal  # on global time
                sound_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sound_4.stopped')
                # update status
                sound_4.status = FINISHED
                sound_4.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[sound_4]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            alphabet_6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in alphabet_6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "alphabet_6" ---
    for thisComponent in alphabet_6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for alphabet_6
    alphabet_6.tStop = globalClock.getTime(format='float')
    alphabet_6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('alphabet_6.stopped', alphabet_6.tStop)
    # check responses
    if key_resp_memory_test_3.keys in ['', [], None]:  # No response was made
        key_resp_memory_test_3.keys = None
    thisExp.addData('key_resp_memory_test_3.keys',key_resp_memory_test_3.keys)
    if key_resp_memory_test_3.keys != None:  # we had a response
        thisExp.addData('key_resp_memory_test_3.rt', key_resp_memory_test_3.rt)
        thisExp.addData('key_resp_memory_test_3.duration', key_resp_memory_test_3.duration)
    sound_4.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if alphabet_6.maxDurationReached:
        routineTimer.addTime(-alphabet_6.maxDuration)
    elif alphabet_6.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-228.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "finish" ---
    # create an object to store info about Routine finish
    finish = data.Routine(
        name='finish',
        components=[image_6, key_resp],
    )
    finish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for finish
    finish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    finish.tStart = globalClock.getTime(format='float')
    finish.status = STARTED
    thisExp.addData('finish.started', finish.tStart)
    finish.maxDuration = None
    # keep track of which components have finished
    finishComponents = finish.components
    for thisComponent in finish.components:
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
    
    # --- Run Routine "finish" ---
    finish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_6* updates
        
        # if image_6 is starting this frame...
        if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_6.frameNStart = frameN  # exact frame index
            image_6.tStart = t  # local t and not account for scr refresh
            image_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_6.started')
            # update status
            image_6.status = STARTED
            image_6.setAutoDraw(True)
        
        # if image_6 is active this frame...
        if image_6.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            finish.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in finish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "finish" ---
    for thisComponent in finish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for finish
    finish.tStop = globalClock.getTime(format='float')
    finish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('finish.stopped', finish.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


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


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
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
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
