import org.apache.commons.io.FilenameUtils;
import wav.WavFile;
import wav.WavFileException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

public class SpectrographLoader implements Iterator<BufferedImage> {

    private static final String WAV_EXTENSION = "wav";
    private static final String SPECTROGRAPH_EXTENSION = "png";
    private static final String SPECTROGRAMS_FOLDER = "spectrograms";

    private FftUtil fftUtil;
    private FftUtil.Strategy strategy;
    private File parentDirectory;

    private Map<String, File> spectrographMap = new HashMap<>();
    private Map<String, File> wavMap = new HashMap<>();

    private Iterator<String> iterator;

    public SpectrographLoader(File parentDirectory, FftUtil fftUtil, FftUtil.Strategy strategy) {
        this.fftUtil = fftUtil;
        this.strategy = strategy;
        this.parentDirectory = parentDirectory;

        if (parentDirectory.listFiles() == null) {
            throw new RuntimeException("No files in directory " + parentDirectory.getName());
        }

        for (File file : parentDirectory.listFiles()) {
            if (FilenameUtils.getExtension(file.getName()).equalsIgnoreCase(WAV_EXTENSION)) {
                wavMap.put(file.getName(), file);
            }

            if (file.getName().equalsIgnoreCase(SPECTROGRAMS_FOLDER)) {
                for (File specFile : file.listFiles()) {
                    spectrographMap.put(FilenameUtils.removeExtension(specFile.getName()), specFile);
                }
            }
        }

        if (wavMap.keySet().size() == 0) {
            throw new RuntimeException("No wav files in directory " + parentDirectory.getName());
        }

        iterator = wavMap.keySet().iterator();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public BufferedImage next() {
        String wavKey = iterator.next();
        if (spectrographMap.get(wavKey) == null) {
            System.out.println("Spect file for " + wavKey + " not found, doing FFT");
            WavFile wavFile;
            double[][] data;
            try {
                wavFile = WavFile.openWavFile(wavMap.get(wavKey));
                data = fftUtil.convertToSpectrogram(wavFile, strategy);
                File spectrogramFile = saveSpectrograph(data, wavKey);
                spectrographMap.put(wavKey, spectrogramFile);
                return loadSpectrograph(spectrogramFile);
            } catch (IOException | WavFileException e) {
                throw new RuntimeException(e.getMessage() + "\n at " + wavKey);
            }
        } else {
            try {
                System.out.println("Spect file for " + wavKey + " found, skipping FFT");
                return loadSpectrograph(spectrographMap.get(wavKey));
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }
        }
    }

    private File saveSpectrograph(double[][] dataMatrix, String fileName) throws IOException {

        double maxValue = getMaximumArrayValue(dataMatrix);
        double numPerStep = maxValue / (double) Integer.MAX_VALUE;

        BufferedImage theImage = new BufferedImage(dataMatrix.length, dataMatrix[0].length, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < dataMatrix.length; x++) {
            for (int y = 0; y < dataMatrix[0].length; y++) {
                int rgbValue = (int) (dataMatrix[x][y] / numPerStep);
                if(rgbValue < Integer.MAX_VALUE / 1500) {
                    rgbValue = 0;
                }
                theImage.setRGB(x, y, rgbValue);
            }
        }

        File targetFile = new File(parentDirectory.getAbsolutePath() + "/" + SPECTROGRAMS_FOLDER + "/" + fileName + "." + SPECTROGRAPH_EXTENSION);
        targetFile.mkdirs();
        ImageIO.write(theImage, "png", targetFile);
        return targetFile;
    }

    private BufferedImage loadSpectrograph(File file) throws IOException {
        return ImageIO.read(file);
    }


    private double getMaximumArrayValue(double[][] dataMatrix) {
        double max = 0;
        for (int i = 0; i < dataMatrix.length; i++) {
            for (int j = 0; j < dataMatrix[i].length; j++) {
                if (dataMatrix[i][j] > max) {
                    max = dataMatrix[i][j];
                }
            }
        }
        System.out.println("Max Value: " + max);
        return max;
    }

}
