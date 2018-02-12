import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import wav.WavFile;
import wav.WavFileException;

import java.io.IOException;

public class FftUtil {

    public static final int REQUIRED_BIT_DEPTH = 16;
    public static final int REQUIRED_SAMPLE_RATE = 44100;

    private FastFourierTransformer fastFourierTransformer;
    private int frequencyBinCount;
    private int fftFrameSize;
    private int fftFramesPerClip;
    private int audioFramesPerClip;

    public FftUtil(int requiredClipLengthSeconds, int fftFrameSize) {
        this.fastFourierTransformer = new FastFourierTransformer(DftNormalization.STANDARD);

        this.fftFrameSize = fftFrameSize;
        this.frequencyBinCount = fftFrameSize;


        this.fftFramesPerClip = requiredClipLengthSeconds * (REQUIRED_SAMPLE_RATE / fftFrameSize);
        this.audioFramesPerClip = requiredClipLengthSeconds * REQUIRED_SAMPLE_RATE;
    }

    public double[][] convertToSpectrogram(WavFile wavFile, Strategy strategy) throws WavFileException, IOException {
        long startTime = System.currentTimeMillis();
        if (wavFile.getValidBits() != REQUIRED_BIT_DEPTH) {
            throw new RuntimeException("Invalid bit depth. Use " + REQUIRED_BIT_DEPTH);
        }
        if (wavFile.getSampleRate() != REQUIRED_SAMPLE_RATE) {
            throw new RuntimeException("Invalid sample rate.  Use " + REQUIRED_SAMPLE_RATE);
        }
        int[][] multiChannelSampleBuffer = new int[wavFile.getNumChannels()][fftFrameSize];
        double[] monoChannelSampleBuffer = new double[fftFrameSize];
        double[][] spectrogram = new double[fftFramesPerClip][frequencyBinCount / 8];
        int framesRead = 0;
        int currentFFTFrame = 0;
        LOOP:
        do {
            framesRead += wavFile.readFrames(multiChannelSampleBuffer, fftFrameSize);

            switch (strategy) {
                case START:
                    if (framesRead >= fftFramesPerClip * fftFrameSize) {
                        break LOOP;
                    }
                    break;
                case MIDDLE:
                    if (framesRead < getMiddleClipFramesCount(wavFile)) {
                        continue LOOP;
                    } else if (framesRead >= getMiddleClipFramesCount(wavFile) + (fftFramesPerClip * fftFrameSize)) {
                        break LOOP;
                    }
                    break;
                case END:
                    if (framesRead < getEndClipFramesCount(wavFile)) {
                        continue LOOP;
                    }
                    break;
            }

            monoChannelSampleBuffer = sum16BitToMono(multiChannelSampleBuffer, monoChannelSampleBuffer);
            monoChannelSampleBuffer = toHanningWindow(monoChannelSampleBuffer);
            Complex[] frameSpectrogram = fastFourierTransformer.transform(monoChannelSampleBuffer, TransformType.FORWARD);

            for (int i = (frameSpectrogram.length / 8) * 7; i < frameSpectrogram.length; i++) {
                spectrogram[currentFFTFrame][i - (frameSpectrogram.length / 8) * 7] = Math.abs(frameSpectrogram[i].getReal()
                                * frameSpectrogram[i].getReal() + frameSpectrogram[i].getImaginary()
                                * frameSpectrogram[i].getImaginary());
            }
            currentFFTFrame++;
        } while (framesRead != 0);

        System.out.println("FFT done in " + (System.currentTimeMillis() - startTime) + " ms");
        return spectrogram;
    }

    private double[] sum16BitToMono(int[][] stereoSamples, double[] monoPCM) {
        for (int i = 0; i < stereoSamples[0].length; i++) {
            if(stereoSamples.length > 1) {
                monoPCM[i] = stereoSamples[0][i] + stereoSamples[1][i] / 2;
            } else {
                monoPCM[i] = stereoSamples[0][i];
            }
        }
        return monoPCM;
    }

    private double[] toHanningWindow(double[] recordedData) {

        // iterate until the last line of the data buffer
        for (int n = 1; n < recordedData.length; n++) {
            // reduce unnecessarily performed frequency part of each and every frequency
            recordedData[n] *= 0.5 * (1 - Math.cos((2 * Math.PI * n)
                    / (recordedData.length - 1)));
        }
        // return modified buffer to the FFT function
        return recordedData;
    }

    private int getMiddleClipFramesCount(WavFile wavFile) {
        return (int) ((wavFile.getNumFrames() / 2) - (audioFramesPerClip / 2));
    }

    private int getEndClipFramesCount(WavFile wavFile) {
        return (int) (wavFile.getNumFrames() - audioFramesPerClip);
    }

    public int getMatrixHeight() {
        return frequencyBinCount / 8;
    }

    public int getMatrixWidth() {
        return fftFramesPerClip;
    }

    public enum Strategy {
        START, MIDDLE, END
    }


}
